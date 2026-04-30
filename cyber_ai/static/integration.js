/**
 * NeuroSent — Backend Integration Layer
 * 
 * Connects the existing dashboard frontend to the FastAPI backend
 * via WebSocket for real-time ML threat predictions.
 * 
 * This file is injected into index.html by the FastAPI server.
 * It is idempotent — safe to load multiple times.
 */

(function NeuroSentIntegration() {
  'use strict';

  // Guard: prevent double initialization
  if (window.__NEUROSENT_INTEGRATED__) return;
  window.__NEUROSENT_INTEGRATED__ = true;

  // ─── Configuration ───
  const WS_URL = `ws://${window.location.host}/ws/threats`;
  const RECONNECT_DELAY = 3000;
  const MAX_FEED_ROWS = 50;
  const MAX_ALERTS = 3;
  const ARC_FADE_MS = 4000;

  // ─── State ───
  let ws = null;
  let reconnectTimer = null;
  let feedCount = 0;

  // ─── Country → approximate map coordinates (percentage of map area) ───
  const COUNTRY_COORDS = {
    'China':         { x: 78, y: 35 },
    'Russia':        { x: 65, y: 22 },
    'United States': { x: 20, y: 35 },
    'Brazil':        { x: 30, y: 65 },
    'North Korea':   { x: 82, y: 34 },
    'Iran':          { x: 60, y: 38 },
    'Germany':       { x: 52, y: 28 },
    'India':         { x: 68, y: 45 },
    'Vietnam':       { x: 76, y: 48 },
    'Indonesia':     { x: 80, y: 58 },
  };

  // Target location (defending network — US East Coast)
  const TARGET_COORD = { x: 25, y: 38 };

  // Severity → CSS class mapping
  const SEV_CLASS_MAP = {
    'CRITICAL': 'sev-crit',
    'HIGH':     'sev-high',
    'MEDIUM':   'sev-med',
    'LOW':      'sev-low',
  };

  // Severity → alert panel class mapping
  const ALERT_CLASS_MAP = {
    'CRITICAL': 'crit',
    'HIGH':     'high',
    'MEDIUM':   'med',
    'LOW':      'med',
  };

  // Threat type display names
  const THREAT_DISPLAY = {
    'ddos':          'DDoS Flood',
    'port_scan':     'Port Scan',
    'brute_force':   'Brute Force',
    'sql_injection': 'SQL Injection',
    'malware_c2':    'Malware C2',
    'zero_day':      'Zero-Day',
    'normal':        'Normal',
  };

  // Threat type → alert title
  const ALERT_TITLES = {
    'ddos':          '⚠ DDoS Attack Detected',
    'port_scan':     '◆ Port Scan Activity',
    'brute_force':   '⚠ Brute Force Login',
    'sql_injection': '⚠ SQL Injection Attempt',
    'malware_c2':    '☠ Malware C2 Communication',
    'zero_day':      '🔴 Zero-Day Exploit',
    'normal':        '✓ Normal Traffic',
  };

  // Attack type → bar text content matching (case-insensitive partial match)
  const ATTACK_BAR_MAP = {
    'ddos':          'DDoS',
    'port_scan':     'Port Scan',
    'brute_force':   'Brute Force',
    'sql_injection': 'SQL Injection',
    'malware_c2':    'Malware C2',
    'zero_day':      'Zero-Day',
  };


  // ═══════════════════════════════════════════════════════
  // WEBSOCKET CONNECTION
  // ═══════════════════════════════════════════════════════

  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      ws = new WebSocket(WS_URL);
    } catch (e) {
      console.warn('[NeuroSent] WebSocket creation failed:', e);
      scheduleReconnect();
      return;
    }

    ws.onopen = function () {
      console.log('[NeuroSent] WebSocket connected');
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    ws.onmessage = function (event) {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'threat' && msg.data) {
          updateThreatFeed(msg.data);
          if (msg.data.threat_detected) {
            updateAlerts(msg.data);
            addAttackArc(msg.data);
          }
        } else if (msg.type === 'stats' && msg.data) {
          updateStats(msg.data);
        }
      } catch (e) {
        console.warn('[NeuroSent] Failed to parse message:', e);
      }
    };

    ws.onclose = function () {
      console.log('[NeuroSent] WebSocket disconnected');
      scheduleReconnect();
    };

    ws.onerror = function (err) {
      console.warn('[NeuroSent] WebSocket error:', err);
      ws.close();
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      connectWebSocket();
    }, RECONNECT_DELAY);
  }


  // ═══════════════════════════════════════════════════════
  // COUNTER ANIMATION
  // ═══════════════════════════════════════════════════════

  function animateValue(element, newValue, duration) {
    if (!element) return;
    duration = duration || 600;

    const text = element.textContent.replace(/,/g, '').replace(/%/g, '');
    const startVal = parseFloat(text) || 0;
    const endVal = typeof newValue === 'number' ? newValue : parseFloat(String(newValue).replace(/,/g, '')) || 0;
    const isPercent = String(newValue).includes('%');
    const isFloat = !Number.isInteger(endVal);

    if (Math.abs(startVal - endVal) < 0.01) return;

    const startTime = performance.now();

    function step(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = startVal + (endVal - startVal) * eased;

      if (isFloat) {
        element.textContent = current.toFixed(1) + (isPercent ? '%' : '');
      } else {
        element.textContent = Math.round(current).toLocaleString();
      }

      if (progress < 1) {
        requestAnimationFrame(step);
      }
    }

    requestAnimationFrame(step);
  }


  // ═══════════════════════════════════════════════════════
  // UPDATE STATS
  // ═══════════════════════════════════════════════════════

  function updateStats(data) {
    // Stat cards
    const statThreats = document.getElementById('stat-threats');
    const statActive = document.getElementById('stat-active');
    const sbIncidents = document.getElementById('sb-incidents');
    const sbEps = document.getElementById('sb-eps');

    if (statThreats && data.threats_today != null) {
      animateValue(statThreats, data.threats_today);
    }
    if (statActive && data.active_alerts != null) {
      animateValue(statActive, data.active_alerts);
    }
    if (sbIncidents && data.active_alerts != null) {
      animateValue(sbIncidents, data.active_alerts);
    }
    if (sbEps && data.events_per_second != null) {
      animateValue(sbEps, data.events_per_second);
    }

    // Update attack classification bars
    if (data.attack_breakdown) {
      updateAttackBars(data.attack_breakdown);
    }
  }

  function updateAttackBars(breakdown) {
    // Calculate total for percentages
    const total = Object.values(breakdown).reduce(function (s, v) { return s + v; }, 0) || 1;

    // Find all attack-item elements in the Attack Classification section
    const attackItems = document.querySelectorAll('.attack-list .attack-item');
    // The first .attack-list section is "Attack Classification"
    const sections = document.querySelectorAll('.section-label');
    let attackSection = null;

    for (let i = 0; i < sections.length; i++) {
      if (sections[i].textContent.trim().toLowerCase().includes('attack classification')) {
        // The attack-list is the next sibling element
        attackSection = sections[i].parentElement;
        break;
      }
    }

    if (!attackSection) return;

    const items = attackSection.querySelectorAll('.attack-item');
    items.forEach(function (item) {
      const headerSpan = item.querySelector('.attack-header span:first-child');
      const pctSpan = item.querySelector('.attack-pct');
      const barFill = item.querySelector('.bar-fill');

      if (!headerSpan || !pctSpan || !barFill) return;

      const text = headerSpan.textContent.trim();

      // Match attack type
      for (const [key, displayName] of Object.entries(ATTACK_BAR_MAP)) {
        if (text.toLowerCase().includes(displayName.toLowerCase())) {
          const count = breakdown[key] || 0;
          const pct = Math.round((count / total) * 100) || 0;
          pctSpan.textContent = pct + '%';
          barFill.style.width = Math.min(pct, 100) + '%';
          break;
        }
      }
    });
  }


  // ═══════════════════════════════════════════════════════
  // UPDATE THREAT FEED
  // ═══════════════════════════════════════════════════════

  function updateThreatFeed(data) {
    const feedScroll = document.getElementById('feedScroll');
    if (!feedScroll) return;

    // Pause scroll animation while inserting
    feedScroll.style.animation = 'none';

    const sevClass = SEV_CLASS_MAP[data.severity] || 'sev-low';
    const typeName = THREAT_DISPLAY[data.threat_type] || data.threat_type;
    const country = data.country || 'Unknown';
    const sourceIp = data.source_ip || '0.0.0.0';
    const timestamp = data.timestamp || new Date().toISOString().split('T')[1].split('.')[0];

    const row = document.createElement('div');
    row.className = 'feed-item';
    row.innerHTML =
      '<span class="feed-time">' + timestamp + '</span>' +
      '<span class="feed-ip">' + sourceIp + '</span>' +
      '<span class="feed-type">' + typeName + '</span>' +
      '<span class="feed-country">' + country + '</span>' +
      '<span class="severity ' + sevClass + '">' + data.severity + '</span>';

    // Add fade-in effect
    row.style.opacity = '0';
    row.style.transform = 'translateX(-10px)';
    row.style.transition = 'opacity 0.3s ease, transform 0.3s ease';

    feedScroll.insertBefore(row, feedScroll.firstChild);

    // Trigger animation
    requestAnimationFrame(function () {
      row.style.opacity = '1';
      row.style.transform = 'translateX(0)';
    });

    feedCount++;

    // Keep max rows
    while (feedScroll.children.length > MAX_FEED_ROWS) {
      feedScroll.removeChild(feedScroll.lastChild);
    }

    // Restart scroll animation after a brief pause
    setTimeout(function () {
      feedScroll.style.animation = '';
    }, 500);
  }


  // ═══════════════════════════════════════════════════════
  // UPDATE ACTIVE ALERTS
  // ═══════════════════════════════════════════════════════

  function updateAlerts(data) {
    if (!data.threat_detected) return;

    // Find the Active Alerts section
    const sections = document.querySelectorAll('.section-label');
    let alertsContainer = null;

    for (let i = 0; i < sections.length; i++) {
      if (sections[i].textContent.trim().toLowerCase().includes('active alerts')) {
        alertsContainer = sections[i].parentElement;
        break;
      }
    }

    if (!alertsContainer) return;

    const alertClass = ALERT_CLASS_MAP[data.severity] || 'med';
    const title = ALERT_TITLES[data.threat_type] || '⚠ Threat Detected';
    const meta = (data.source_ip || '0.0.0.0') + ' → ' + (data.destination_ip || '10.0.0.1') + ' // ' + (data.timestamp || '--:--:--');

    // Build SHAP bars HTML
    let shapHtml = '<div class="alert-shap">Why flagged:</div>';
    if (data.shap_features && data.shap_features.length > 0) {
      data.shap_features.forEach(function (sf) {
        const widthPct = Math.round((sf.value || 0) * 100);
        shapHtml +=
          '<div class="shap-bar">' +
            '<div class="shap-lbl">' + (sf.feature || '') + '</div>' +
            '<div class="shap-track">' +
              '<div class="shap-fill" style="width:' + widthPct + '%"></div>' +
            '</div>' +
            '<span style="font-size:9px;color:#00e5ff;font-family:Share Tech Mono">' + (sf.value || 0).toFixed(2) + '</span>' +
          '</div>';
      });
    }

    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert-item ' + alertClass;
    alertDiv.innerHTML =
      '<div class="alert-title">' + title + '</div>' +
      '<div class="alert-meta">' + meta + '</div>' +
      shapHtml;

    // Fade-in animation
    alertDiv.style.opacity = '0';
    alertDiv.style.transform = 'translateY(-8px)';
    alertDiv.style.transition = 'opacity 0.4s ease, transform 0.4s ease';

    // Find the section-label and insert after it
    const sectionLabel = alertsContainer.querySelector('.section-label');
    if (sectionLabel && sectionLabel.nextSibling) {
      alertsContainer.insertBefore(alertDiv, sectionLabel.nextSibling);
    } else {
      alertsContainer.appendChild(alertDiv);
    }

    // Trigger animation
    requestAnimationFrame(function () {
      alertDiv.style.opacity = '1';
      alertDiv.style.transform = 'translateY(0)';
    });

    // Keep max alerts
    const allAlerts = alertsContainer.querySelectorAll('.alert-item');
    while (allAlerts.length > MAX_ALERTS) {
      const oldest = allAlerts[allAlerts.length - 1];
      oldest.style.opacity = '0';
      oldest.style.transform = 'translateY(8px)';
      setTimeout(function () {
        if (oldest.parentNode) oldest.parentNode.removeChild(oldest);
      }, 300);
      break; // remove one at a time to avoid DOM issues
    }
  }


  // ═══════════════════════════════════════════════════════
  // ATTACK ARC ANIMATION ON MAP
  // ═══════════════════════════════════════════════════════

  function addAttackArc(data) {
    const mapArea = document.getElementById('mapArea');
    if (!mapArea) return;

    const country = data.country || '';
    const sourceCoord = COUNTRY_COORDS[country];
    if (!sourceCoord) return;

    // Find or create the SVG overlay for arcs
    let arcSvg = mapArea.querySelector('.attack-arc-overlay');
    if (!arcSvg) {
      arcSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      arcSvg.setAttribute('class', 'attack-arc-overlay');
      arcSvg.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:10;';
      arcSvg.setAttribute('viewBox', '0 0 100 100');
      arcSvg.setAttribute('preserveAspectRatio', 'none');
      mapArea.appendChild(arcSvg);
    }

    // Determine arc color from severity
    let arcColor = '#00e5ff';
    if (data.severity === 'CRITICAL') arcColor = '#ff1744';
    else if (data.severity === 'HIGH') arcColor = '#ff6d00';
    else if (data.severity === 'MEDIUM') arcColor = '#ffd600';

    const sx = sourceCoord.x;
    const sy = sourceCoord.y;
    const tx = TARGET_COORD.x;
    const ty = TARGET_COORD.y;

    // Compute a curved arc control point
    const mx = (sx + tx) / 2;
    const my = Math.min(sy, ty) - 15 - Math.random() * 10;

    const pathD = 'M ' + sx + ' ' + sy + ' Q ' + mx + ' ' + my + ' ' + tx + ' ' + ty;

    // Create arc path
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', pathD);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', arcColor);
    path.setAttribute('stroke-width', '0.4');
    path.setAttribute('opacity', '0.8');
    path.setAttribute('stroke-dasharray', '2 1.5');

    // Animate dash offset for flow effect
    const animateDash = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
    animateDash.setAttribute('attributeName', 'stroke-dashoffset');
    animateDash.setAttribute('from', '0');
    animateDash.setAttribute('to', '-10');
    animateDash.setAttribute('dur', '1s');
    animateDash.setAttribute('repeatCount', 'indefinite');
    path.appendChild(animateDash);

    // Animate opacity fade out
    const animateFade = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
    animateFade.setAttribute('attributeName', 'opacity');
    animateFade.setAttribute('from', '0.8');
    animateFade.setAttribute('to', '0');
    animateFade.setAttribute('begin', (ARC_FADE_MS / 1000 - 1) + 's');
    animateFade.setAttribute('dur', '1s');
    animateFade.setAttribute('fill', 'freeze');
    path.appendChild(animateFade);

    arcSvg.appendChild(path);

    // Add a small ping circle at source
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', String(sx));
    circle.setAttribute('cy', String(sy));
    circle.setAttribute('r', '0.8');
    circle.setAttribute('fill', arcColor);
    circle.setAttribute('opacity', '0.9');

    const animateR = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
    animateR.setAttribute('attributeName', 'r');
    animateR.setAttribute('from', '0.5');
    animateR.setAttribute('to', '3');
    animateR.setAttribute('dur', '1.5s');
    animateR.setAttribute('repeatCount', '2');
    circle.appendChild(animateR);

    const animateOp = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
    animateOp.setAttribute('attributeName', 'opacity');
    animateOp.setAttribute('from', '0.9');
    animateOp.setAttribute('to', '0');
    animateOp.setAttribute('dur', '1.5s');
    animateOp.setAttribute('repeatCount', '2');
    circle.appendChild(animateOp);

    arcSvg.appendChild(circle);

    // Clean up after fade
    setTimeout(function () {
      if (path.parentNode) path.parentNode.removeChild(path);
      if (circle.parentNode) circle.parentNode.removeChild(circle);
    }, ARC_FADE_MS);
  }


  // ═══════════════════════════════════════════════════════
  // INITIALIZATION
  // ═══════════════════════════════════════════════════════

  function init() {
    console.log('[NeuroSent] Integration layer initializing ...');

    // Clear static placeholder feed items and replace with empty container
    const feedScroll = document.getElementById('feedScroll');
    if (feedScroll && feedScroll.children.length > 0) {
      // Keep existing placeholder items but they'll be pushed down by new ones
    }

    // Connect WebSocket
    connectWebSocket();

    console.log('[NeuroSent] Integration layer ready ✓');
  }

  // Wait for DOM
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();

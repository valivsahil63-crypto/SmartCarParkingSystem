/**
 * dashboard.js — Smart Parking System frontend logic
 * Polls /api/stats and /api/plates every second.
 */

const POLL_INTERVAL = 1000; // ms

let plateData = [];

// ── Clock ──────────────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const pad = n => String(n).padStart(2, "0");
  document.getElementById("clockDisplay").textContent =
    `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
}
setInterval(updateClock, 1000);
updateClock();

// ── Occupancy Gauge ────────────────────────────────────────────────────────────
function updateGauge(free, occupied, total) {
  const pct     = total > 0 ? Math.round((occupied / total) * 100) : 0;
  const arcLen  = 251;   // approximate arc length of the semicircle path
  const offset  = arcLen - (pct / 100) * arcLen;

  document.getElementById("gaugeArc").style.strokeDashoffset = offset;
  document.getElementById("gaugePct").textContent            = `${pct}%`;

  // Colour the arc based on occupancy
  const arc = document.getElementById("gaugeArc");
  if (pct >= 80) {
    arc.setAttribute("stroke", "url(#occupiedGrad)");
  } else if (pct >= 50) {
    arc.setAttribute("stroke", "#ffab40");
  } else {
    arc.setAttribute("stroke", "url(#freeGrad)");
  }
}

// ── Plates Renderer ────────────────────────────────────────────────────────────
function renderPlates(plates) {
  const list = document.getElementById("plateList");
  if (!plates || plates.length === 0) {
    list.innerHTML = '<div class="plate-empty">No plates detected yet…</div>';
    return;
  }
  list.innerHTML = plates.map(p => `
    <div class="plate-entry">
      <span class="plate-number">${p.plate}</span>
      <span class="plate-conf">${Math.round(p.confidence * 100)}%</span>
      <span class="plate-time">${p.time || "--"}</span>
    </div>
  `).join("");
}

// ── Animate number change ──────────────────────────────────────────────────────
function animateValue(el, newVal) {
  const current = parseInt(el.textContent) || 0;
  if (current === newVal) return;
  el.style.transform = "scale(1.2)";
  el.style.transition = "transform 0.15s ease";
  el.textContent = newVal;
  setTimeout(() => {
    el.style.transform = "scale(1)";
  }, 150);
}

// ── Stats Polling ──────────────────────────────────────────────────────────────
async function pollStats() {
  try {
    const res  = await fetch("/api/stats");
    const data = await res.json();

    // Status indicator
    document.getElementById("statusDot").className  = "status-dot active";
    document.getElementById("statusText").textContent = "Live";
    document.getElementById("fpsPill").textContent    = `${data.fps} FPS`;

    // Slot stats
    animateValue(document.getElementById("freeCount"),     data.free);
    animateValue(document.getElementById("occupiedCount"), data.occupied);
    animateValue(document.getElementById("totalCount"),    data.total_slots);
    updateGauge(data.free, data.occupied, data.total_slots);

    // Vehicle counter
    animateValue(document.getElementById("enteredCount"), data.entered);
    animateValue(document.getElementById("exitedCount"),  data.exited);
    const inside = Math.max(0, data.entered - data.exited);
    document.getElementById("insideCount").textContent = inside;

  } catch (err) {
    document.getElementById("statusDot").className  = "status-dot error";
    document.getElementById("statusText").textContent = "Disconnected";
  }
}

async function pollPlates() {
  try {
    const res    = await fetch("/api/plates");
    const plates = await res.json();
    renderPlates(plates);
  } catch (_) {}
}

// ── Controls ───────────────────────────────────────────────────────────────────
document.getElementById("setSourceBtn").addEventListener("click", async () => {
  const src = document.getElementById("sourceInput").value.trim();
  const btn = document.getElementById("setSourceBtn");
  btn.textContent = "Setting…";
  try {
    await fetch("/api/set_source", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ source: src }),
    });
    // Force video feed refresh
    const img = document.getElementById("videoFeed");
    img.src   = "/video_feed?" + Date.now();
    btn.textContent = "✓ Done";
  } catch (e) {
    btn.textContent = "Error";
  }
  setTimeout(() => btn.textContent = "Set Source", 2000);
});

document.getElementById("reloadSlotsBtn").addEventListener("click", async () => {
  const btn = document.getElementById("reloadSlotsBtn");
  btn.textContent = "Loading…";
  try {
    const res  = await fetch("/api/reload_slots", { method: "POST" });
    const data = await res.json();
    btn.textContent = `✓ ${data.slots} Slots`;
  } catch {
    btn.textContent = "Error";
  }
  setTimeout(() => btn.textContent = "↺ Reload Slots", 2500);
});

function clearPlates() {
  document.getElementById("plateList").innerHTML =
    '<div class="plate-empty">Cleared. New plates will appear here…</div>';
}

// ── Start Polling ──────────────────────────────────────────────────────────────
setInterval(pollStats,  POLL_INTERVAL);
setInterval(pollPlates, POLL_INTERVAL * 2);
pollStats();
pollPlates();

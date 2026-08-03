/** Scrub demo cream MP4 against open-loop policy mechinterp. */

const $ = (id) => document.getElementById(id);

const SERVO = ["dx", "dy", "dz", "dax", "day", "daz", "grip"];
const GROUPS = ["fusion", "drift", "trm", "tqsa", "relational", "planner", "corrector"];

async function main() {
  const res = await fetch("data/demo_cream/latest.json", { cache: "no-store" });
  if (!res.ok) {
    $("status").textContent =
      "No pack yet. Run: .venv/bin/python paper/activation_webapp/generate_demo_trace.py";
    return;
  }
  const pack = await res.json();
  const tr = await fetch(`data/demo_cream/${pack.trace}`, { cache: "no-store" });
  if (!tr.ok) {
    $("status").textContent = "latest.json found but trace.json missing.";
    return;
  }
  const trace = await tr.json();
  const primaryName = (pack.primary && pack.primary.checkpoint || "primary").split("/").pop();
  const compareName = pack.compare
    ? (pack.compare.checkpoint || "compare").split("/").pop()
    : null;

  $("status").textContent =
    `${pack.n_ticks} ticks · ${primaryName}` +
    (compareName ? ` vs ${compareName}` : "") +
    ` · ${pack.control || "open_loop"}`;

  const player = $("player");
  player.addEventListener("error", () => {
    const err = player.error;
    $("status").textContent =
      `Video failed (code ${err && err.code}). Re-encode H.264 baseline or hard-refresh.`;
  });
  player.src = `data/demo_cream/${pack.video}?v=${Date.now()}`;
  player.load();

  drawSpark(trace, primaryName, compareName);

  const paint = () => {
    const fps = trace.fps || pack.fps || 30;
    const stride = trace.stride || 1;
    const frame = Math.floor(player.currentTime * fps);
    const tickIdx = Math.min(
      trace.ticks.length - 1,
      Math.max(0, Math.floor(frame / stride)),
    );
    showTick(trace, tickIdx, primaryName, compareName);
  };
  player.addEventListener("timeupdate", paint);
  player.addEventListener("seeked", paint);
  paint();
}

function drawSpark(trace, primaryName, compareName) {
  const canvas = $("grip-spark");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ebe4d6";
  ctx.fillRect(0, 0, w, h);
  // zero line
  const mid = h / 2;
  ctx.strokeStyle = "#b0a691";
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(w, mid);
  ctx.stroke();

  const grips = trace.ticks.map((t) => t.grip);
  const cgrips = trace.ticks.map((t) => (t.compare ? t.compare.grip : null));
  const maxAbs = Math.max(0.2, ...grips.map(Math.abs),
    ...cgrips.filter((x) => x != null).map(Math.abs));

  const plot = (arr, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    arr.forEach((g, i) => {
      if (g == null) return;
      const x = (i / Math.max(1, arr.length - 1)) * w;
      const y = mid - (g / maxAbs) * (h * 0.42);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  plot(grips, "#b45309");
  if (compareName) plot(cgrips, "#1d4e89");

  $("spark-caption").textContent =
    `Grip over demo (orange=${primaryName}` +
    (compareName ? `, blue=${compareName}` : "") +
    `). Negative ≈ close.`;
}

function showTick(trace, i, primaryName, compareName) {
  const tick = trace.ticks[i];
  if (!tick) return;
  const names = trace.servo_names || SERVO;

  $("choice-meta").innerHTML =
    `<span class="pill">${tick.is_real ? "REAL" : "dream"}</span>` +
    `<span>t=${tick.t} · frame ${tick.frame}</span>` +
    `<span>trust ${Number(tick.trust).toFixed(2)}</span>` +
    (tick.action_l1 != null
      ? `<span class="pill warn">ΔL1 ${Number(tick.action_l1).toFixed(2)}</span>`
      : "") +
    (tick.src_conf != null ? `<span>src ${Number(tick.src_conf).toFixed(2)}</span>` : "");

  const bars = $("servo-bars");
  bars.innerHTML = "";
  const a = tick.action || [];
  const b = (tick.compare && tick.compare.action) || null;
  const maxAbs = Math.max(0.15, ...a.map(Math.abs), ...(b || []).map(Math.abs));
  names.forEach((name, j) => {
    const row = document.createElement("div");
    row.className = "sbar";
    const v = a[j] || 0;
    const vc = b ? (b[j] || 0) : null;
    const pct = (x) => 50 + 50 * (x / maxAbs);
    const fill = (x, cls) => {
      const p = pct(x);
      const left = Math.min(50, p);
      const width = Math.abs(p - 50);
      return `<div class="fill ${cls}" style="left:${left}%;width:${width}%"></div>`;
    };
    row.innerHTML =
      `<span>${name}</span>` +
      `<div class="track"><div class="zero"></div>${fill(v, "")}` +
      (vc != null ? fill(vc, "cmp") : "") +
      `</div>` +
      `<span class="gval">${v.toFixed(2)}</span>`;
    bars.appendChild(row);
  });

  const gc = $("grip-compare");
  gc.innerHTML =
    `<div class="grip-card primary"><div class="lbl">${primaryName}</div>` +
    `<div class="val">${Number(tick.grip).toFixed(3)}</div></div>` +
    (tick.compare
      ? `<div class="grip-card compare"><div class="lbl">${compareName}</div>` +
        `<div class="val">${Number(tick.compare.grip).toFixed(3)}</div></div>`
      : "");

  $("live-meta").innerHTML =
    `<span class="pill">${tick.is_real ? "REAL" : "dream"}</span>` +
    `<span>plan‖ ${Number(tick.plan_norm || 0).toFixed(2)}</span>` +
    (tick.src_center
      ? `<span>uv (${tick.src_center.map((x) => Number(x).toFixed(2)).join(", ")})</span>`
      : "");

  const ge = tick.group_e || {};
  const maxE = Math.max(1e-6, ...GROUPS.map((g) => ge[g] || 0));
  const gb = $("group-bars");
  gb.innerHTML = "";
  for (const g of GROUPS) {
    if (!(g in ge)) continue;
    const v = ge[g] || 0;
    const row = document.createElement("div");
    row.className = "gbar";
    row.innerHTML =
      `<span class="gname">${g}</span>` +
      `<div class="track"><div class="fill" style="width:${(100 * v / maxE).toFixed(1)}%"></div></div>` +
      `<span class="gval">${v.toFixed(1)}</span>`;
    gb.appendChild(row);
  }

  const acts = Object.entries(tick.acts || {})
    .filter(([, s]) => s && typeof s.l2 === "number")
    .sort((a, b) => b[1].l2 - a[1].l2)
    .slice(0, 12);
  const list = $("hot-list");
  list.innerHTML = "";
  for (const [id, s] of acts) {
    const li = document.createElement("li");
    li.innerHTML = `<code>${id}</code><span>ℓ₂ ${s.l2.toFixed(2)}</span>`;
    list.appendChild(li);
  }
}

main().catch((e) => {
  $("status").textContent = String(e);
});

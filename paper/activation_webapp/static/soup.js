/** Play saved 1080p LIBERO soup success + live activation scrub. */

const $ = (id) => document.getElementById(id);

const FLOW = [
  { id: "see", title: "Eyes", blurb: "Wrist camera + YOLO finds the soup can and the basket." },
  { id: "stack", title: "Learned stack", blurb: "Fusion → HRM → TRM → relational → planner still run every tick (activations below)." },
  { id: "ibvs", title: "Phased IBVS (owns the arm)", blurb: "A zero-training state machine replaces the planner’s action: servo → align → grasp → lift → transport → release using detections + proprio." },
  { id: "success", title: "Soup in basket", blurb: "Assisted success — not an unaided policy score." },
];

const GROUP_ORDER = ["fusion", "drift", "trm", "tqsa", "relational", "planner", "corrector", "ibvs"];

async function main() {
  const res = await fetch("data/soup_success/latest.json", { cache: "no-store" });
  if (!res.ok) {
    $("status").textContent =
      "No soup pack yet. On the box run: python -m eval.record_soup_angles … then scp data/soup_success/ here.";
    buildFlow(null);
    return;
  }
  const pack = await res.json();
  const aided = pack.unaided === false || pack.control === "assisted_phased_ibvs"
    || (pack.config || "").startsWith("soup_v");
  $("status").textContent =
    `SUCCESS · init ${pack.init_index} · ${pack.n_steps} steps · ` +
    `${pack.film_w}×${pack.film_h} · ${pack.config}` +
    (aided ? " · ASSISTED (PhasedIBVS owns actions)" : " · unaided") +
    ` · ${pack.task}`;

  const cams = pack.cameras || Object.keys(pack.videos || {});
  const labels = pack.labels || {};
  const bar = $("angles");
  bar.innerHTML = "";
  cams.forEach((cam, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = labels[cam] || cam;
    b.className = i === 0 ? "active" : "";
    b.onclick = () => {
      for (const x of bar.querySelectorAll("button")) x.classList.remove("active");
      b.classList.add("active");
      $("player").src = `data/soup_success/${pack.videos[cam]}`;
      $("player").play().catch(() => {});
    };
    bar.appendChild(b);
  });

  const player = $("player");
  player.addEventListener("error", () => {
    $("status").textContent =
      "Video failed to decode (need H.264). Re-encode with ffmpeg libx264.";
  });
  if (cams[0]) {
    player.src = `data/soup_success/${pack.videos[cams[0]]}`;
    player.load();
  }

  let trace = null;
  if (pack.trace) {
    try {
      const tr = await fetch(`data/soup_success/${pack.trace}`, { cache: "no-store" });
      if (tr.ok) trace = await tr.json();
    } catch (_) { /* ignore */ }
  }
  if (trace && Array.isArray(trace.ticks) && trace.ticks.length) {
    const fps = pack.fps || 30;
    const paint = () => showTick(trace, Math.min(
      trace.ticks.length - 1,
      Math.max(0, Math.floor(player.currentTime * fps)),
    ));
    player.addEventListener("timeupdate", paint);
    player.addEventListener("seeked", paint);
    paint();
  } else {
    $("live-meta").textContent =
      "No activation trace in this pack — refilm with the hooked recorder.";
  }

  buildFlow(pack);
}

function showTick(trace, i) {
  const tick = trace.ticks[i];
  if (!tick) return;
  const phase = tick.phase || "—";
  const real = tick.is_real ? "REAL" : "dream";
  $("live-meta").innerHTML =
    `<span class="pill">${real}</span>` +
    `<span class="pill phase">${phase}</span>` +
    `<span>t=${tick.t} · trust ${Number(tick.trust).toFixed(2)}</span>` +
    `<span>plan‖ ${Number(tick.plan_norm || 0).toFixed(2)}</span>` +
    (tick.src_conf != null ? `<span>src ${Number(tick.src_conf).toFixed(2)}</span>` : "");

  const ge = tick.group_e || {};
  const maxE = Math.max(1e-6, ...GROUP_ORDER.map((g) => ge[g] || 0));
  const bars = $("group-bars");
  bars.innerHTML = "";
  for (const g of GROUP_ORDER) {
    if (!(g in ge) && g !== "ibvs") continue;
    const v = ge[g] || 0;
    const row = document.createElement("div");
    row.className = "gbar";
    row.innerHTML =
      `<span class="gname">${g}</span>` +
      `<div class="track"><div class="fill" style="width:${(100 * v / maxE).toFixed(1)}%"></div></div>` +
      `<span class="gval">${v.toFixed(1)}</span>`;
    bars.appendChild(row);
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

function buildFlow(pack) {
  const root = $("flow");
  root.innerHTML = "";
  FLOW.forEach((node, i) => {
    const el = document.createElement("div");
    el.className = "flow-node";
    el.innerHTML = `<div class="num">${i + 1}</div>
      <div><h3>${node.title}</h3><p>${node.blurb}</p></div>`;
    root.appendChild(el);
    if (i < FLOW.length - 1) {
      const arrow = document.createElement("div");
      arrow.className = "flow-arrow";
      arrow.textContent = "↓";
      root.appendChild(arrow);
    }
  });
  if (pack) {
    const note = document.createElement("p");
    note.className = "flow-note";
    note.textContent =
      "Phased IBVS = Image-Based Visual Servoing with an explicit phase machine. " +
      "It is NOT the learned policy controlling the arm on this video — the net still " +
      "runs (see Live model), but emitted deltas come from the servo/grasp/place automaton.";
    root.appendChild(note);
  }
}

main().catch((e) => {
  $("status").textContent = String(e);
});

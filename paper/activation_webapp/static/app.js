import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);

let TRACE = null;
let tick = 0;
let playing = false;
let selected = "planner";
let lastTs = 0;
let raf = 0;

const groupOrder = ["fusion", "drift", "trm", "tqsa", "relational", "planner", "corrector"];

async function main() {
  const res = await fetch("data/trace.json", { cache: "no-store" });
  if (!res.ok) {
    $("metaLine").textContent = "Missing data/trace.json — run generate_trace.py";
    return;
  }
  TRACE = await res.json();
  const m = TRACE.meta;
  $("metaLine").textContent =
    `${m.checkpoint} [${m.weights || "?"}] · ${m.ticks} ticks @ ${m.tick_hz} Hz (${m.duration_s}s) · ` +
    `${m.n_modules} modules · ${m.n_weight_tensors} weight tensors`;
  $("scrub").max = String(m.ticks - 1);
  $("contribNote").textContent = m.note;
  buildDag();
  initSim();
  wire();
  selectModule(selected);
  renderTick(0);
  loop(0);
}

/** Pipeline nodes for the flow DAG (plan § Architecture DAG). */
const FLOW_NODES = [
  "perception", "fusion", "drift", "trm", "corrector", "tqsa", "relational", "planner", "plan",
];
/** Edges that carry planner attribution thickness when available. */
const FLOW_EDGES = [
  ["perception", "fusion"],
  ["fusion", "trm"],
  ["drift", "trm"],
  ["trm", "corrector"],
  ["corrector", "planner"],
  ["tqsa", "planner"],
  ["relational", "planner"],
  ["fusion", "planner"],
  ["drift", "planner"],
  ["planner", "plan"],
];
const EDGE_SENS = {
  "fusion->planner": "fused",
  "fusion->trm": "fused",
  "drift->trm": "state_delta",
  "drift->planner": "state_delta",
  "trm->corrector": "next_emb",
  "corrector->planner": "next_emb",
  "tqsa->planner": "spatial",
  "relational->planner": "relational",
};

function wire() {
  $("btnPlay").onclick = () => {
    playing = !playing;
    $("btnPlay").textContent = playing ? "Pause" : "Play";
  };
  $("btnStep").onclick = () => {
    playing = false;
    $("btnPlay").textContent = "Play";
    setTick(Math.min(tick + 1, TRACE.meta.ticks - 1));
  };
  $("scrub").oninput = (e) => setTick(Number(e.target.value));
  $("modFilter").oninput = () => buildDag($("modFilter").value.trim().toLowerCase());
}

function setTick(t) {
  tick = t;
  $("scrub").value = String(t);
  renderTick(t);
}

function loop(ts) {
  raf = requestAnimationFrame(loop);
  if (!playing || !TRACE) return;
  const speed = Number($("speed").value);
  const dt = (ts - lastTs) / 1000;
  if (dt < 1 / (TRACE.meta.tick_hz * speed)) return;
  lastTs = ts;
  if (tick >= TRACE.meta.ticks - 1) {
    playing = false;
    $("btnPlay").textContent = "Play";
    return;
  }
  setTick(tick + 1);
}

function buildDag(filter = "") {
  const root = $("dag");
  root.innerHTML = "";
  const byGroup = Object.fromEntries(groupOrder.map((g) => [g, []]));
  for (const mod of TRACE.modules) {
    if (!byGroup[mod.group]) byGroup[mod.group] = [];
    byGroup[mod.group].push(mod);
  }
  for (const g of Object.keys(byGroup)) {
    const mods = byGroup[g].filter((m) => {
      if (!filter) return true;
      return m.id.toLowerCase().includes(filter) || m.type.toLowerCase().includes(filter);
    });
    if (!mods.length) continue;
    const gh = document.createElement("div");
    gh.className = "group";
    gh.textContent = `${g} · ${mods.length}`;
    root.appendChild(gh);
    // Sort: parents first-ish by depth, then name
    mods.sort((a, b) => a.id.localeCompare(b.id));
    for (const mod of mods) {
      const el = document.createElement("div");
      el.className = `mod${mod.leaf ? " leaf" : ""}${mod.id === selected ? " active" : ""}`;
      el.dataset.id = mod.id;
      const depth = (mod.name.match(/\./g) || []).length;
      el.innerHTML = `
        <span class="name" style="padding-left:${depth * 8}px">${mod.name || mod.id}</span>
        <span class="type">${mod.type}</span>
        <span class="bar"><i></i></span>`;
      el.onclick = () => selectModule(mod.id);
      root.appendChild(el);
    }
  }
}

function selectModule(id) {
  selected = id;
  for (const el of document.querySelectorAll(".mod")) {
    el.classList.toggle("active", el.dataset.id === id);
  }
  const mod = TRACE.modules.find((m) => m.id === id);
  $("inspectTitle").textContent = id;
  $("inspectType").textContent = mod ? `${mod.type} · ${mod.params} params` : "—";
  drawSpark();
  renderTick(tick);
}

function renderTick(t) {
  const row = TRACE.ticks[t];
  if (!row) return;
  $("clock").textContent = `${row.t_s.toFixed(2)}s`;
  const badge = $("simBadge");
  badge.textContent = row.is_real ? "REAL" : "dream";
  badge.className = `badge ${row.is_real ? "real" : "dream"}`;

  // module energy bars
  let maxL2 = 1e-6;
  for (const s of Object.values(row.acts)) maxL2 = Math.max(maxL2, s.l2 || 0);
  for (const el of document.querySelectorAll(".mod")) {
    const s = row.acts[el.dataset.id];
    const i = el.querySelector("i");
    const pct = s ? Math.min(100, (100 * (s.l2 || 0)) / maxL2) : 0;
    i.style.width = `${pct}%`;
    i.style.background = s && s.sat > 0.35 ? "var(--danger)" : "var(--accent)";
  }

  // inspector
  const act = row.acts[selected];
  if (act) {
    $("inspectStats").textContent =
      `t=${row.t}  l2=${act.l2}  mean=${act.mean}  absmax=${act.absmax}\n` +
      `sat=${act.sat}  n=${act.n}  trust=${row.trust}  latent‖‖=${row.latent_l2}  next‖‖=${row.next_l2}`;
    const bars = $("topBars");
    bars.innerHTML = "";
    const tops = act.top || [];
    const peak = Math.max(1e-6, ...tops.map(Math.abs));
    for (const v of tops) {
      const b = document.createElement("div");
      b.className = "b on";
      b.style.height = `${Math.max(4, (56 * Math.abs(v)) / peak)}px`;
      b.title = String(v);
      bars.appendChild(b);
    }
  } else {
    $("inspectStats").textContent = "no activation this tick (module idle)";
    $("topBars").innerHTML = "";
  }
  drawSpark();
  drawHeat($("planHeat"), row.plan, true);
  drawHeat($("fusedHeat"), row.fused || row.fused_row_l2?.map((v) => [v]) || [], false);
  drawContrib(row);
  drawFlowDag(row);
  updateSim(row);
}

function drawFlowDag(row) {
  const c = $("flowDag");
  if (!c) return;
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.fillStyle = "#10151a";
  ctx.fillRect(0, 0, w, h);

  const nodes = TRACE.meta.flow || FLOW_NODES;
  const n = nodes.length;
  const pad = 28;
  const xs = nodes.map((_, i) => pad + (i * (w - 2 * pad)) / Math.max(1, n - 1));
  const y = h * 0.52;

  const ge = row.group_e || {};
  const maxE = Math.max(1e-6, ...Object.values(ge), ge.planner || 0, 1);
  // perception / plan synthetic energy
  const energy = {
    perception: row.is_real ? maxE * 0.85 : maxE * 0.15,
    plan: (row.plan0 || []).reduce((a, v) => a + Math.abs(v), 0) || 0.2,
    ...ge,
  };

  const sens = Object.keys(row.sens || {}).length ? row.sens : (TRACE.sens_mean || {});
  const maxS = Math.max(1e-6, ...Object.values(sens), 0.01);

  // edges
  for (const [a, b] of FLOW_EDGES) {
    const ia = nodes.indexOf(a), ib = nodes.indexOf(b);
    if (ia < 0 || ib < 0) continue;
    const key = `${a}->${b}`;
    const sk = EDGE_SENS[key];
    const thick = sk && sens[sk] != null
      ? 1 + 8 * (sens[sk] / maxS)
      : 1.5;
    ctx.strokeStyle = sk && sens[sk] > 0.05 ? "#d6a45a" : "#3a4550";
    ctx.lineWidth = thick;
    ctx.beginPath();
    ctx.moveTo(xs[ia], y);
    const mid = (xs[ia] + xs[ib]) / 2;
    ctx.quadraticCurveTo(mid, y - 28, xs[ib], y);
    ctx.stroke();
  }

  // nodes
  nodes.forEach((name, i) => {
    const e = energy[name] || 0;
    const t = Math.min(1, e / maxE);
    const r = 10 + 8 * t;
    ctx.beginPath();
    ctx.fillStyle = `rgb(${40 + Math.round(170 * t)},${50 + Math.round(90 * t)},${40 + Math.round(40 * (1 - t))})`;
    ctx.strokeStyle = name === selected.split(".")[0] ? "#e6ecef" : "#2a333c";
    ctx.lineWidth = 2;
    ctx.arc(xs[i], y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#8a97a3";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText(name, xs[i], y + 28);
  });
}

function drawSpark() {
  const c = $("spark");
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.fillStyle = "#10151a";
  ctx.fillRect(0, 0, w, h);
  if (!TRACE) return;
  const series = TRACE.ticks.map((r) => (r.acts[selected]?.l2) || 0);
  const peak = Math.max(1e-6, ...series);
  ctx.strokeStyle = "#2a333c";
  ctx.beginPath();
  for (let i = 0; i < 5; i++) {
    const y = (h * i) / 4;
    ctx.moveTo(0, y); ctx.lineTo(w, y);
  }
  ctx.stroke();
  // real tick markers
  ctx.fillStyle = "rgba(92,184,138,0.15)";
  for (let i = 0; i < TRACE.ticks.length; i++) {
    if (!TRACE.ticks[i].is_real) continue;
    const x = (i / (TRACE.ticks.length - 1)) * w;
    ctx.fillRect(x, 0, 2, h);
  }
  ctx.strokeStyle = "#d6a45a";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < series.length; i++) {
    const x = (i / (series.length - 1)) * (w - 2) + 1;
    const y = h - 4 - ((h - 8) * series[i]) / peak;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  // playhead
  const x = (tick / (TRACE.ticks.length - 1)) * w;
  ctx.strokeStyle = "#e6ecef";
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
}

function drawHeat(canvas, mat, signed) {
  const ctx = canvas.getContext("2d");
  const rows = mat?.length || 0;
  const cols = rows ? (Array.isArray(mat[0]) ? mat[0].length : 1) : 0;
  ctx.fillStyle = "#10151a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!rows || !cols) return;
  const cw = canvas.width / cols, ch = canvas.height / rows;
  let peak = 1e-6;
  for (const r of mat) {
    const vals = Array.isArray(r) ? r : [r];
    for (const v of vals) peak = Math.max(peak, Math.abs(v));
  }
  for (let i = 0; i < rows; i++) {
    const vals = Array.isArray(mat[i]) ? mat[i] : [mat[i]];
    for (let j = 0; j < vals.length; j++) {
      const v = vals[j] / peak;
      if (signed) {
        const t = (v + 1) / 2;
        ctx.fillStyle = `rgb(${Math.round(40 + 180 * t)},${Math.round(50 + 90 * (1 - Math.abs(v)))},${Math.round(40 + 160 * (1 - t))})`;
      } else {
        const t = Math.abs(v);
        ctx.fillStyle = `rgb(${Math.round(30 + 200 * t)},${Math.round(40 + 140 * t)},${Math.round(50 + 40 * t)})`;
      }
      ctx.fillRect(j * cw + 1, i * ch + 1, cw - 2, ch - 2);
    }
  }
}

function drawContrib(row) {
  const c = $("contrib");
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.fillStyle = "#10151a";
  ctx.fillRect(0, 0, w, h);
  const sens = Object.keys(row.sens || {}).length ? row.sens : TRACE.sens_mean;
  const keys = Object.keys(sens || {}).sort((a, b) => sens[b] - sens[a]);
  const ge = $("groupEnergy");
  ge.innerHTML = "";
  for (const [g, v] of Object.entries(row.group_e || {})) {
    const s = document.createElement("span");
    s.textContent = `${g} Σl2=${v}`;
    ge.appendChild(s);
  }
  if (!keys.length) {
    ctx.fillStyle = "#8a97a3";
    ctx.font = "12px monospace";
    ctx.fillText("sensitivity available on REAL ticks", 16, 40);
    return;
  }
  const max = Math.max(...keys.map((k) => sens[k]), 1e-6);
  const barH = Math.min(22, (h - 30) / keys.length - 4);
  keys.forEach((k, i) => {
    const y = 16 + i * (barH + 6);
    const bw = ((w - 140) * sens[k]) / max;
    ctx.fillStyle = "#2a333c";
    ctx.fillRect(120, y, w - 140, barH);
    ctx.fillStyle = "#d6a45a";
    ctx.fillRect(120, y, bw, barH);
    ctx.fillStyle = "#e6ecef";
    ctx.font = "11px monospace";
    ctx.fillText(k, 8, y + barH - 4);
    ctx.fillStyle = "#8a97a3";
    ctx.fillText(sens[k].toFixed(3), w - 48, y + barH - 4);
  });
}

/* ---------------- Three.js LIBERO-like sim ---------------- */

let renderer, scene, camera, controls, armParts = {}, objMeshes = {}, trailLine;
let gripL, gripR;

function initSim() {
  const canvas = $("sim");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.85;
  renderer.shadowMap.enabled = true;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x12161a);
  scene.fog = new THREE.Fog(0x12161a, 2.2, 5.5);

  camera = new THREE.PerspectiveCamera(42, 1, 0.01, 20);
  camera.position.set(0.72, 0.55, 0.85);

  controls = new OrbitControls(camera, canvas);
  controls.target.set(0.0, 0.06, 0.0);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.48;

  // Lights — keep contrast high so table/objects read on dark UI
  const hemi = new THREE.HemisphereLight(0xc8d2dc, 0x3a2a1c, 0.55);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xfff0d8, 1.1);
  key.position.set(0.7, 1.4, 0.55);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.near = 0.1;
  key.shadow.camera.far = 5;
  key.shadow.camera.left = -1.0;
  key.shadow.camera.right = 1.0;
  key.shadow.camera.top = 1.0;
  key.shadow.camera.bottom = -1.0;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x8aa4c8, 0.28);
  fill.position.set(-0.6, 0.7, -0.2);
  scene.add(fill);

  // Room / table (LIBERO kitchen-ish)
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(6, 6),
    new THREE.MeshStandardMaterial({ color: 0x2c3036, roughness: 0.92 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.42;
  floor.receiveShadow = true;
  scene.add(floor);

  const wallMat = new THREE.MeshStandardMaterial({ color: 0x8a8378, roughness: 0.95 });
  const wall = new THREE.Mesh(new THREE.BoxGeometry(2.4, 1.4, 0.06), wallMat);
  wall.position.set(0, 0.35, -0.62);
  wall.receiveShadow = true;
  scene.add(wall);

  const cabMat = new THREE.MeshStandardMaterial({ color: 0x6e6356, roughness: 0.8 });
  for (const x of [-0.9, 0.9]) {
    const cab = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.65, 0.32), cabMat);
    cab.position.set(x, -0.08, -0.4);
    cab.castShadow = true; cab.receiveShadow = true;
    scene.add(cab);
  }

  const table = new THREE.Mesh(
    new THREE.BoxGeometry(0.75, 0.05, 0.58),
    new THREE.MeshStandardMaterial({ color: 0x5c3d24, roughness: 0.5, metalness: 0.08 })
  );
  table.position.set(0, 0.0, 0.02);
  table.castShadow = true; table.receiveShadow = true;
  scene.add(table);

  const matTop = new THREE.MeshStandardMaterial({ color: 0x8b6844, roughness: 0.35 });
  const top = new THREE.Mesh(new THREE.BoxGeometry(0.78, 0.012, 0.6), matTop);
  top.position.set(0, 0.031, 0.02);
  top.receiveShadow = true;
  scene.add(top);

  const legMat = new THREE.MeshStandardMaterial({ color: 0x2a221c, roughness: 0.7 });
  for (const [lx, lz] of [[-0.32, -0.22], [0.32, -0.22], [-0.32, 0.24], [0.32, 0.24]]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.42, 0.035), legMat);
    leg.position.set(lx, -0.21, lz);
    leg.castShadow = true;
    scene.add(leg);
  }

  // Robot pedestal + arm
  const ped = new THREE.Mesh(
    new THREE.CylinderGeometry(0.08, 0.1, 0.12, 32),
    new THREE.MeshStandardMaterial({ color: 0x2c3138, metalness: 0.4, roughness: 0.45 })
  );
  ped.position.set(0, 0.06, -0.28);
  ped.castShadow = true;
  scene.add(ped);

  const white = new THREE.MeshStandardMaterial({ color: 0xe8eaed, metalness: 0.25, roughness: 0.35 });
  const black = new THREE.MeshStandardMaterial({ color: 0x22262c, metalness: 0.5, roughness: 0.4 });

  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.06, 24), black);
  base.position.set(0, 0.15, -0.28);
  scene.add(base);
  armParts.base = base;

  const link1 = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.18, 0.05), white);
  link1.position.set(0, 0.26, -0.28);
  link1.castShadow = true;
  scene.add(link1);
  armParts.link1 = link1;

  const link2 = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.22), white);
  link2.castShadow = true;
  scene.add(link2);
  armParts.link2 = link2;

  const wrist = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.045, 0.05), black);
  wrist.castShadow = true;
  scene.add(wrist);
  armParts.wrist = wrist;

  gripL = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.04, 0.01), black);
  gripR = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.04, 0.01), black);
  scene.add(gripL, gripR);

  // Objects from first tick scene template
  const sc0 = TRACE.ticks[0].scene;
  for (const [name, o] of Object.entries(sc0.objects)) {
    const geo = name === "basket"
      ? new THREE.BoxGeometry(o.size[0], o.size[2], o.size[1])
      : new THREE.BoxGeometry(o.size[0], o.size[2], o.size[1]);
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(o.color),
      roughness: name === "basket" ? 0.85 : 0.45,
      metalness: name === "milk" ? 0.15 : 0.05,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true; mesh.receiveShadow = true;
    mesh.position.set(o.pos[0], o.pos[2], o.pos[1]);
    scene.add(mesh);
    objMeshes[name] = mesh;

    if (name === "basket") {
      // hollow look: darker inner
      const inner = new THREE.Mesh(
        new THREE.BoxGeometry(o.size[0] * 0.85, o.size[2] * 0.7, o.size[1] * 0.85),
        new THREE.MeshStandardMaterial({ color: 0x3a2a18, roughness: 1 })
      );
      inner.position.y = 0.01;
      mesh.add(inner);
    }
  }

  // Trail
  const trailGeo = new THREE.BufferGeometry();
  const positions = new Float32Array(900 * 3);
  trailGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  trailGeo.setDrawRange(0, 0);
  trailLine = new THREE.Line(
    trailGeo,
    new THREE.LineBasicMaterial({ color: 0xd6a45a, transparent: true, opacity: 0.75 })
  );
  scene.add(trailLine);

  const grid = new THREE.GridHelper(1.0, 20, 0x4a5560, 0x2a333c);
  grid.position.y = 0.038;
  scene.add(grid);

  const onResize = () => {
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(2, rect.width);
    const h = Math.max(2, rect.height);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  window.addEventListener("resize", onResize);
  onResize();
}

function placeArm(eef, grip) {
  const target = new THREE.Vector3(eef[0], eef[2], eef[1]);
  const shoulder = new THREE.Vector3(0, 0.26, -0.28);
  armParts.link1.position.copy(shoulder);
  // aim link1 toward mid
  const mid = shoulder.clone().lerp(target, 0.45);
  armParts.link1.position.copy(mid);
  armParts.link1.lookAt(target);
  armParts.link1.rotateX(Math.PI / 2);

  armParts.link2.position.copy(shoulder.clone().lerp(target, 0.75));
  armParts.link2.lookAt(target);

  armParts.wrist.position.copy(target);
  const open = 0.008 + 0.018 * Math.max(0, grip);
  gripL.position.set(target.x - open, target.y - 0.02, target.z);
  gripR.position.set(target.x + open, target.y - 0.02, target.z);
  // trust-ish opacity via emissive
}

function updateSim(row) {
  if (!renderer) return;
  const sc = row.scene;
  placeArm(sc.eef, sc.grip);
  for (const [name, o] of Object.entries(sc.objects)) {
    const mesh = objMeshes[name];
    if (!mesh) continue;
    mesh.position.set(o.pos[0], o.pos[2], o.pos[1]);
  }
  // trail up to current tick
  const pos = trailLine.geometry.attributes.position.array;
  const n = Math.min(tick + 1, TRACE.trail.length);
  for (let i = 0; i < n; i++) {
    const p = TRACE.trail[i];
    pos[i * 3] = p[0];
    pos[i * 3 + 1] = p[2];
    pos[i * 3 + 2] = p[1];
  }
  trailLine.geometry.setDrawRange(0, n);
  trailLine.geometry.attributes.position.needsUpdate = true;

  // camera slight follow
  controls.target.lerp(new THREE.Vector3(sc.eef[0] * 0.3, 0.08, sc.eef[1] * 0.3), 0.05);
  controls.update();
  renderer.render(scene, camera);

  $("simReadout").textContent =
    `eef=(${sc.eef.map((v) => v.toFixed(3)).join(", ")})  grip=${sc.grip.toFixed(2)}  ` +
    `dist_src=${sc.dist_src.toFixed(3)}  held=${sc.held ?? "—"}  ` +
    `trust=${row.trust.toFixed(3)}  src_conf=${row.src_conf ?? "—"}  ` +
    `success=${sc.success}`;
}

main().catch((e) => {
  console.error(e);
  $("metaLine").textContent = String(e);
});

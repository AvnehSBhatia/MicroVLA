/** Dual view: cream demo video + local Three.js arm from model actions. */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const SERVO = ["dx", "dy", "dz", "dax", "day", "daz", "grip"];

let TRACE = null;
let TRAJ = null;
let tick = 0;
let playing = false;
let lastTs = 0;
let raf = 0;

let renderer, scene, camera, controls;
let modelArm = {};
let demoGhost = {};
let modelTrail, demoTrail;
let creamMesh;

async function main() {
  const [packRes, trRes, tjRes] = await Promise.all([
    fetch("data/demo_cream/latest.json", { cache: "no-store" }),
    fetch("data/demo_cream/trace.json", { cache: "no-store" }),
    fetch("data/demo_cream/traj.json", { cache: "no-store" }),
  ]);
  if (!packRes.ok || !trRes.ok || !tjRes.ok) {
    $("status").textContent =
      "Missing pack. Run generate_demo_trace.py then rebuild traj.";
    return;
  }
  const pack = await packRes.json();
  TRACE = await trRes.json();
  TRAJ = await tjRes.json();
  const n = Math.min(TRACE.ticks.length, TRAJ.n);
  $("scrub").max = String(n - 1);
  $("status").textContent =
    `${n} ticks · model=${(pack.primary?.checkpoint || "").split("/").pop()} · ` +
    `gain=${TRAJ.gain} · local sim (not MuJoCo)`;

  const player = $("player");
  player.addEventListener("error", () => {
    $("status").textContent = "Video failed — hard-refresh or re-encode H.264.";
  });
  player.src = `data/demo_cream/${pack.video}?v=2`;
  player.load();

  initSim();
  drawXy();
  wire(player);
  setTick(0);
  loop(0);
}

function wire(player) {
  $("btnPlay").onclick = () => {
    playing = !playing;
    $("btnPlay").textContent = playing ? "Pause" : "Play";
    if (playing && $("lockVideo").checked) player.play().catch(() => {});
    else player.pause();
  };
  $("btnStep").onclick = () => {
    playing = false;
    $("btnPlay").textContent = "Play";
    player.pause();
    setTick(Math.min(tick + 1, TRAJ.n - 1));
  };
  $("scrub").oninput = (e) => {
    playing = false;
    $("btnPlay").textContent = "Play";
    setTick(Number(e.target.value));
  };
  player.addEventListener("seeked", () => {
    if (!$("lockVideo").checked) return;
    const fps = TRACE.fps || 30;
    const frame = Math.floor(player.currentTime * fps);
    setTick(Math.min(TRAJ.n - 1, Math.max(0, frame)));
  });
}

function setTick(t) {
  tick = t;
  $("scrub").value = String(t);
  const fps = TRACE.fps || 30;
  $("clock").textContent = `${(t / fps).toFixed(2)}s`;
  if ($("lockVideo").checked) {
    const player = $("player");
    const want = t / fps;
    if (Math.abs(player.currentTime - want) > 0.08) {
      player.currentTime = want;
    }
  }
  renderTick(t);
}

function loop(ts) {
  raf = requestAnimationFrame(loop);
  if (!playing) {
    lastTs = ts;
    return;
  }
  const fps = TRACE.fps || 30;
  const speed = 1;
  if (ts - lastTs > (1000 / fps) / speed) {
    lastTs = ts;
    if (tick >= TRAJ.n - 1) {
      playing = false;
      $("btnPlay").textContent = "Play";
      $("player").pause();
      return;
    }
    setTick(tick + 1);
  }
}

function initSim() {
  const canvas = $("sim");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  const w = canvas.clientWidth || 640;
  const h = canvas.clientHeight || 360;
  renderer.setSize(w, h, false);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0a09);
  scene.fog = new THREE.Fog(0x0b0a09, 2.4, 6);

  camera = new THREE.PerspectiveCamera(42, w / h, 0.05, 20);
  camera.position.set(0.55, 0.55, 0.7);
  controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0.08, 0);
  controls.enableDamping = true;

  scene.add(new THREE.HemisphereLight(0xf3efe6, 0x2a241c, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(0.6, 1.2, 0.4);
  scene.add(key);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(1.6, 1.6),
    new THREE.MeshStandardMaterial({ color: 0x3a2f24, roughness: 0.9 }),
  );
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);

  const table = new THREE.Mesh(
    new THREE.BoxGeometry(0.7, 0.04, 0.7),
    new THREE.MeshStandardMaterial({ color: 0x6b5340, roughness: 0.85 }),
  );
  table.position.y = 0.02;
  scene.add(table);

  creamMesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.06, 0.035, 0.045),
    new THREE.MeshStandardMaterial({ color: 0x3b6ea5 }),
  );
  creamMesh.position.set(0.05, 0.04, -0.02);
  scene.add(creamMesh);

  const basket = new THREE.Mesh(
    new THREE.CylinderGeometry(0.07, 0.06, 0.05, 16),
    new THREE.MeshStandardMaterial({ color: 0xb8a07a, wireframe: false }),
  );
  basket.position.set(0.22, 0.05, 0.18);
  scene.add(basket);

  modelArm = buildArm(0xe07a3a);
  demoGhost = buildArm(0x5b8def, 0.35);

  modelTrail = makeTrail(0xe07a3a);
  demoTrail = makeTrail(0x5b8def);
  scene.add(modelTrail, demoTrail);

  window.addEventListener("resize", () => {
    const ww = canvas.clientWidth || 640;
    const hh = canvas.clientHeight || 360;
    camera.aspect = ww / hh;
    camera.updateProjectionMatrix();
    renderer.setSize(ww, hh, false);
  });
}

function buildArm(color, opacity = 1) {
  const mat = new THREE.MeshStandardMaterial({
    color,
    transparent: opacity < 1,
    opacity,
    roughness: 0.45,
  });
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 0.03, 20), mat);
  base.position.y = 0.05;
  scene.add(base);
  const link1 = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.18, 0.03), mat);
  scene.add(link1);
  const link2 = new THREE.Mesh(new THREE.BoxGeometry(0.025, 0.16, 0.025), mat);
  scene.add(link2);
  const wrist = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 12), mat);
  scene.add(wrist);
  const gripL = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.04, 0.012), mat);
  const gripR = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.04, 0.012), mat);
  scene.add(gripL, gripR);
  return { base, link1, link2, wrist, gripL, gripR };
}

function makeTrail(color) {
  const geo = new THREE.BufferGeometry();
  const n = TRAJ.n;
  const pos = new Float32Array(n * 3);
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setDrawRange(0, 0);
  return new THREE.Line(
    geo,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 }),
  );
}

function placeArm(parts, eef, grip) {
  // LIBERO-ish: x right, y forward, z up → three.js x,z,y
  const target = new THREE.Vector3(eef[0], eef[2], eef[1]);
  parts.wrist.position.copy(target);
  parts.link2.position.set(target.x * 0.55, target.y * 0.55 + 0.08, target.z * 0.55);
  parts.link1.position.set(target.x * 0.25, 0.16, target.z * 0.25);
  parts.base.position.set(0, 0.05, 0);
  const open = grip < 0 ? 0.008 : 0.022;
  parts.gripL.position.set(target.x - open, target.y - 0.02, target.z);
  parts.gripR.position.set(target.x + open, target.y - 0.02, target.z);
}

function fillTrail(line, series, upto) {
  const attr = line.geometry.getAttribute("position");
  for (let i = 0; i <= upto; i++) {
    const p = series[i];
    attr.setXYZ(i, p[0], p[2], p[1]);
  }
  attr.needsUpdate = true;
  line.geometry.setDrawRange(0, upto + 1);
}

function renderTick(t) {
  const row = TRACE.ticks[t];
  const me = TRAJ.model_eef[t];
  const de = TRAJ.demo_eef[t];
  const mg = TRAJ.model_grip[t];

  placeArm(modelArm, me, mg);
  placeArm(demoGhost, de, 0.5);
  creamMesh.position.set(de[0], Math.max(0.04, de[2] * 0.3), de[1]);

  fillTrail(modelTrail, TRAJ.model_eef, t);
  fillTrail(demoTrail, TRAJ.demo_eef, t);

  controls.target.lerp(new THREE.Vector3(me[0] * 0.4, 0.08, me[1] * 0.4), 0.08);
  controls.update();
  renderer.render(scene, camera);

  $("simReadout").textContent =
    `model eef=(${me.map((v) => v.toFixed(3)).join(", ")}) grip=${mg.toFixed(2)} · ` +
    `demo proxy=(${de.map((v) => v.toFixed(3)).join(", ")})`;
  const uv = TRAJ.demo_uv[t] || [0.5, 0.5];
  $("demoReadout").textContent =
    `wrist cream UV=(${uv[0].toFixed(2)}, ${uv[1].toFixed(2)}) · frame ${row.frame}`;

  $("tickMeta").innerHTML =
    `<span class="pill model">MODEL</span>` +
    `<span class="pill">${row.is_real ? "REAL" : "dream"}</span>` +
    `<span>trust ${Number(row.trust).toFixed(2)}</span>` +
    `<span class="pill demo">DEMO UV</span>`;

  const a = row.action || [];
  const maxAbs = Math.max(0.15, ...a.map(Math.abs));
  const bars = $("servo-bars");
  bars.innerHTML = "";
  SERVO.forEach((name, j) => {
    const v = a[j] || 0;
    const p = 50 + 50 * (v / maxAbs);
    const left = Math.min(50, p);
    const width = Math.abs(p - 50);
    const el = document.createElement("div");
    el.className = "sbar";
    el.innerHTML =
      `<span>${name}</span>` +
      `<div class="track"><div class="zero"></div>` +
      `<div class="fill" style="left:${left}%;width:${width}%"></div></div>` +
      `<span class="gval">${v.toFixed(2)}</span>`;
    bars.appendChild(el);
  });

  drawXy(t);
}

function drawXy(t = tick) {
  const canvas = $("xyPlot");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#0b0a09";
  ctx.fillRect(0, 0, w, h);
  const map = (p) => {
    // table ±0.35 → canvas
    const x = ((p[0] + 0.35) / 0.7) * w;
    const y = (1 - (p[1] + 0.35) / 0.7) * h;
    return [x, y];
  };
  ctx.strokeStyle = "#3a342c";
  ctx.strokeRect(4, 4, w - 8, h - 8);

  const stroke = (series, color, upto) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= upto; i++) {
      const [x, y] = map(series[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  stroke(TRAJ.demo_eef, "#5b8def", t);
  stroke(TRAJ.model_eef, "#e07a3a", t);

  const [mx, my] = map(TRAJ.model_eef[t]);
  const [dx, dy] = map(TRAJ.demo_eef[t]);
  ctx.fillStyle = "#e07a3a";
  ctx.beginPath();
  ctx.arc(mx, my, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#5b8def";
  ctx.beginPath();
  ctx.arc(dx, dy, 5, 0, Math.PI * 2);
  ctx.fill();
}

main().catch((e) => {
  $("status").textContent = String(e);
});

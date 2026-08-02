/** Play saved 1080p LIBERO soup success angles + a plain-English flowchart. */

const $ = (id) => document.getElementById(id);

const FLOW = [
  { id: "see", title: "Eyes", blurb: "Wrist camera + YOLO finds the soup can and the basket." },
  { id: "fusion", title: "Fusion", blurb: "Packs what it sees into a short scene summary." },
  { id: "hrm", title: "HRM / Drift", blurb: "Tracks how the arm and scene are changing over time." },
  { id: "trm", title: "World model (TRM)", blurb: "Imagines the next moment so the robot can plan ahead between camera frames." },
  { id: "corrector", title: "Trust check", blurb: "Compares dream vs reality; slows down when unsure." },
  { id: "planner", title: "Planner", blurb: "Turns the prediction into a short arm motion." },
  { id: "handeye", title: "Hand–eye assist", blurb: "On this success video: a calibrated servo finishes the last centimeters (assisted run)." },
  { id: "success", title: "Soup in basket", blurb: "Task complete." },
];

async function main() {
  const res = await fetch("data/soup_success/latest.json", { cache: "no-store" });
  if (!res.ok) {
    $("status").textContent =
      "No soup pack yet. On the box run: python -m eval.record_soup_angles … then scp data/soup_success/ here.";
    buildFlow(null);
    return;
  }
  const pack = await res.json();
  $("status").textContent =
    `SUCCESS · init ${pack.init_index} · ${pack.n_steps} steps · ` +
    `${pack.film_w}×${pack.film_h} · ${pack.config} · ${pack.task}`;
  const cams = pack.cameras || Object.keys(pack.videos || {});
  const bar = $("angles");
  bar.innerHTML = "";
  cams.forEach((cam, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = cam;
    b.className = i === 0 ? "active" : "";
    b.onclick = () => {
      for (const x of bar.querySelectorAll("button")) x.classList.remove("active");
      b.classList.add("active");
      $("player").src = `data/soup_success/${pack.videos[cam]}`;
      $("player").play().catch(() => {});
    };
    bar.appendChild(b);
  });
  if (cams[0]) {
    $("player").src = `data/soup_success/${pack.videos[cams[0]]}`;
  }
  buildFlow(pack);
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
      "This recording is an assisted success (hand–eye phase owns the last centimeters). " +
      "It shows the robot finishing alphabet soup → basket so you can watch from four angles.";
    root.appendChild(note);
  }
}

main().catch((e) => {
  $("status").textContent = String(e);
});

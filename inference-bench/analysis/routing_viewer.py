"""Interactive viewer for a routing-count dump (MCORE_ROUTING_COUNTS): pick a step and a MoE layer
(sliders, arrow keys, play) and see the per-expert routed rows and the per-rank loads under two
EP layouts. A single self-contained HTML file (data embedded), so it opens offline in a browser.

    python3 inference-bench/analysis/routing_viewer.py <exp dir> [--max-steps 300] [--ep 8 64]
Writes <exp dir>/routing_viewer.html.
"""

import argparse
import base64
import json
import os
import struct


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--max-steps", type=int, default=300, help="evenly subsample to this many steps")
    ap.add_argument("--ep", type=int, nargs=2, default=[8, 64], help="the two EP layouts to compare")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(os.path.join(args.exp_dir, "routing_counts.jsonl"))]
    # Decode steps at the run's full load: the most common routed-row count.
    rows = max(set(r["routed_rows"] for r in recs), key=lambda n: sum(r["routed_rows"] == n for r in recs))
    steps = [r for r in recs if r["routed_rows"] == rows]
    if len(steps) > args.max_steps:
        stride = len(steps) / args.max_steps
        steps = [steps[int(i * stride)] for i in range(args.max_steps)]
    num_layers = max(i + 1 for r in steps for i, layer in enumerate(r["counts"]) if any(layer))
    num_experts = len(steps[0]["counts"][0])
    flat = [c for r in steps for layer in r["counts"][:num_layers] for c in layer]
    assert max(flat) < 65536
    blob = base64.b64encode(struct.pack(f"<{len(flat)}H", *flat)).decode()
    meta = {
        "steps": [r["step"] for r in steps],
        "times": [round(r["step_time_ms"], 2) for r in steps],
        "layers": num_layers,
        "experts": num_experts,
        "rows": rows,
        "ep": args.ep,
        "title": os.path.basename(os.path.normpath(args.exp_dir)),
    }
    html = TEMPLATE.replace("__META__", json.dumps(meta)).replace("__DATA__", blob)
    path = os.path.join(args.exp_dir, "routing_viewer.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"{path}: {len(steps)} steps x {num_layers} layers x {num_experts} experts")


TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Routing load viewer</title>
<style>
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 16px 24px; color: #222; }
  h1 { font-size: 15px; margin: 0 0 4px; }
  .sub { color: #666; font-size: 12px; margin-bottom: 10px; }
  .controls { display: grid; grid-template-columns: 70px 1fr 260px; gap: 6px 10px; align-items: center;
              font-size: 13px; max-width: 1100px; }
  .controls input[type=range] { width: 100%; }
  button { font-size: 12px; padding: 2px 8px; }
  .panel { max-width: 1100px; margin-top: 14px; }
  .panel h2 { font-size: 13px; margin: 0 0 2px; font-weight: 600; }
  .panel .note { font-size: 12px; color: #555; margin-bottom: 2px; }
  svg { display: block; }
  .tip { position: fixed; pointer-events: none; background: #fff; border: 1px solid #bbb; border-radius: 4px;
         padding: 3px 6px; font-size: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.15); display: none; }
  .axis text { font-size: 10px; fill: #555; }
</style></head>
<body>
<h1 id="title"></h1>
<div class="sub">Each MoE layer waits for its busiest EP rank, so a layer's expert time scales with
max(rank load) / mean. Keys: left/right = step, up/down = layer, space = play.</div>
<div class="controls">
  <span>Step</span><input id="step" type="range" min="0" value="0">
  <span><button id="prevS">&lt;</button> <button id="play">play</button> <button id="nextS">&gt;</button>
        <b id="stepLabel"></b></span>
  <span>Layer</span><input id="layer" type="range" min="0" value="20">
  <span><button id="prevL">&lt;</button> <button id="nextL">&gt;</button> <b id="layerLabel"></b></span>
</div>
<div class="panel"><h2>Busiest rank / mean, every MoE layer of this step (click a layer to select it)</h2>
  <div class="note" id="layerNote"></div><svg id="strip"></svg></div>
<div class="panel"><h2>Routed rows per expert</h2>
  <div class="note" id="expNote"></div><svg id="experts"></svg></div>
<div class="panel"><h2 id="epA"></h2><div class="note" id="epANote"></div><svg id="ranksA"></svg></div>
<div class="panel"><h2 id="epB"></h2><div class="note" id="epBNote"></div><svg id="ranksB"></svg></div>
<div class="tip" id="tip"></div>
<script>
const META = __META__;
const RAW = Uint16Array.from(new Uint16Array(Uint8Array.from(atob("__DATA__"), c => c.charCodeAt(0)).buffer));
const S = META.steps.length, L = META.layers, E = META.experts, [EPA, EPB] = META.ep;
const COL = { A: "#0072B2", B: "#009E73", expert: "#7a7a7a", mean: "#D55E00", grid: "#e3e3e3" };
const NS = "http://www.w3.org/2000/svg";
const W = 1100;
let s = S - 1, l = Math.min(20, L - 1), timer = null;

const counts = (si, li) => RAW.subarray((si * L + li) * E, (si * L + li + 1) * E);
function rankLoads(c, ep) {
  const per = E / ep, out = new Array(ep).fill(0);
  for (let e = 0; e < E; e++) out[Math.floor(e / per)] += c[e];
  return out;
}
const ratio = (loads) => Math.max(...loads) / (loads.reduce((a, b) => a + b, 0) / loads.length);

const tip = document.getElementById("tip");
function hover(el, text) {
  el.addEventListener("mousemove", ev => { tip.style.display = "block"; tip.textContent = text;
    tip.style.left = (ev.clientX + 12) + "px"; tip.style.top = (ev.clientY + 12) + "px"; });
  el.addEventListener("mouseleave", () => { tip.style.display = "none"; });
}
function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(NS, tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(el);
  return el;
}
// Bars with a y axis, a dashed reference line, and per-bar hover text.
function bars(svg, values, { height, color, ref, refLabel, ymax, xlabel, tipText, onClick, highlight, boundaries }) {
  svg.innerHTML = "";
  const m = { l: 44, r: 10, t: 8, b: 26 }, w = W - m.l - m.r, h = height - m.t - m.b;
  svg.setAttribute("width", W); svg.setAttribute("height", height);
  const g = svgEl("g", { transform: `translate(${m.l},${m.t})` }, svg);
  const n = values.length, bw = w / n, y = v => h - (v / ymax) * h;
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = ymax * i / ticks;
    svgEl("line", { x1: 0, x2: w, y1: y(v), y2: y(v), stroke: COL.grid }, g);
    const t = svgEl("text", { x: -6, y: y(v) + 3, "text-anchor": "end", class: "axis" }, g);
    t.textContent = ymax >= 10 ? Math.round(v) : v.toFixed(2);
    t.setAttribute("font-size", "10"); t.setAttribute("fill", "#555");
  }
  values.forEach((v, i) => {
    const gap = n > 100 ? 0 : Math.min(2, bw * 0.15);
    const r = svgEl("rect", { x: i * bw + gap / 2, y: y(v), width: Math.max(bw - gap, 0.5), height: h - y(v),
      fill: highlight === i ? "#CC79A7" : color, rx: n > 100 ? 0 : 2 }, g);
    hover(r, tipText(i, v));
    if (onClick) { r.style.cursor = "pointer"; r.addEventListener("click", () => onClick(i)); }
  });
  (boundaries || []).forEach(b => svgEl("line", { x1: b * bw, x2: b * bw, y1: 0, y2: h, stroke: COL.A,
    "stroke-width": 1.2 }, g));
  if (ref !== undefined) {
    svgEl("line", { x1: 0, x2: w, y1: y(ref), y2: y(ref), stroke: COL.mean, "stroke-dasharray": "5 3" }, g);
    const t = svgEl("text", { x: w - 4, y: y(ref) - 4, "text-anchor": "end", "font-size": 10, fill: COL.mean }, g);
    t.textContent = refLabel;
  }
  const xl = svgEl("text", { x: w / 2, y: h + 20, "text-anchor": "middle", "font-size": 11, fill: "#444" }, g);
  xl.textContent = xlabel;
}

function render() {
  document.getElementById("title").textContent =
    `${META.title}: ${S} decode steps at full load (${META.rows} routed rows/layer), ${L} MoE layers, ${E} experts`;
  document.getElementById("step").value = s;
  document.getElementById("layer").value = l;
  document.getElementById("stepLabel").textContent = `step ${META.steps[s]} (${META.times[s]} ms)`;
  document.getElementById("layerLabel").textContent = `MoE layer ${l}`;

  // Layer strip: this step's busiest/mean for both layouts.
  const rA = [], rB = [];
  for (let li = 0; li < L; li++) { const c = counts(s, li); rA.push(ratio(rankLoads(c, EPA))); rB.push(ratio(rankLoads(c, EPB))); }
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  document.getElementById("layerNote").innerHTML =
    `bars: EP${EPB}; mean over layers EP${EPA} <b>${mean(rA).toFixed(2)}</b>, EP${EPB} <b>${mean(rB).toFixed(2)}</b>`;
  bars(document.getElementById("strip"), rB, { height: 150, color: COL.B, ymax: Math.max(2, ...rB) * 1.08,
    ref: 1, refLabel: "balanced", xlabel: "MoE layer", highlight: l, onClick: i => { l = i; render(); },
    tipText: (i, v) => `layer ${i}: EP${EPB} ${v.toFixed(2)}, EP${EPA} ${rA[i].toFixed(2)}` });

  // Experts.
  const c = Array.from(counts(s, l)), meanE = c.reduce((a, b) => a + b, 0) / E;
  const hot = c.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  document.getElementById("expNote").innerHTML = `mean ${meanE.toFixed(1)} rows/expert; hottest: ` +
    hot.map(([v, i]) => `expert ${i} = ${v} (${(v / meanE).toFixed(1)}x)`).join(", ") +
    `; blue lines: EP${EPA} rank boundaries (${E / EPA} experts each)`;
  const bnd = []; for (let b = E / EPA; b < E; b += E / EPA) bnd.push(b);
  bars(document.getElementById("experts"), c, { height: 210, color: COL.expert, ymax: Math.max(...c) * 1.08,
    ref: meanE, refLabel: "mean", xlabel: "expert id", boundaries: bnd,
    tipText: (i, v) => `expert ${i}: ${v} rows (EP${EPA} rank ${Math.floor(i / (E / EPA))}, EP${EPB} rank ${Math.floor(i / (E / EPB))})` });

  // Ranks under the two layouts.
  [[EPA, "A", "ranksA"], [EPB, "B", "ranksB"]].forEach(([ep, key, id]) => {
    const loads = rankLoads(c, ep), m = loads.reduce((a, b) => a + b, 0) / ep, rel = loads.map(x => x / m);
    const top = rel.indexOf(Math.max(...rel));
    document.getElementById("ep" + key).textContent = `EP${ep}: ${E / ep} experts per rank`;
    document.getElementById("ep" + key + "Note").innerHTML =
      `busiest rank ${top} at <b>${rel[top].toFixed(2)}x</b> the mean, so this layer's expert time is ${rel[top].toFixed(2)}x balanced; ` +
      `min ${Math.min(...rel).toFixed(2)}x`;
    bars(document.getElementById(id), rel, { height: 170, color: COL[key], ymax: Math.max(2, ...rel) * 1.08,
      ref: 1, refLabel: "mean", xlabel: `EP${ep} rank`, highlight: top,
      tipText: (i, v) => `rank ${i}: ${loads[i]} rows, ${v.toFixed(2)}x mean (experts ${i * E / ep}-${(i + 1) * E / ep - 1})` });
  });
}

const stepIn = document.getElementById("step"), layerIn = document.getElementById("layer");
stepIn.max = S - 1; layerIn.max = L - 1;
stepIn.oninput = () => { s = +stepIn.value; render(); };
layerIn.oninput = () => { l = +layerIn.value; render(); };
const move = (ds, dl) => { s = Math.min(S - 1, Math.max(0, s + ds)); l = Math.min(L - 1, Math.max(0, l + dl)); render(); };
document.getElementById("prevS").onclick = () => move(-1, 0);
document.getElementById("nextS").onclick = () => move(1, 0);
document.getElementById("prevL").onclick = () => move(0, -1);
document.getElementById("nextL").onclick = () => move(0, 1);
function togglePlay() {
  const btn = document.getElementById("play");
  if (timer) { clearInterval(timer); timer = null; btn.textContent = "play"; return; }
  btn.textContent = "pause";
  timer = setInterval(() => { s = (s + 1) % S; render(); }, 250);
}
document.getElementById("play").onclick = togglePlay;
document.addEventListener("keydown", ev => {
  if (ev.key === "ArrowLeft") move(-1, 0); else if (ev.key === "ArrowRight") move(1, 0);
  else if (ev.key === "ArrowUp") move(0, 1); else if (ev.key === "ArrowDown") move(0, -1);
  else if (ev.key === " ") { ev.preventDefault(); togglePlay(); } else return;
  ev.preventDefault();
});
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()

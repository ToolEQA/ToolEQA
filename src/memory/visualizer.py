"""Standalone Flask web visualizer for SpatialMemory.

Usage:
    # Option 1: Start server, then point agent to write JSON
    python -m src.memory.visualizer --port 5050

    # Option 2: From code
    from src.memory.visualizer import SpatialMemoryVisualizer
    viz = SpatialMemoryVisualizer(port=5050)
    viz.start()  # runs in background thread
    viz.update(spatial_memory)  # call after each step
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from typing import Any, Optional

DEFAULT_PORT = 5050
DEFAULT_JSON_PATH = ".cache/spatial_memory.json"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spatial Memory Visualizer</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; }
.header {
  padding: 12px 24px; background: #161b22; border-bottom: 1px solid #30363d;
  display: flex; align-items: center; gap: 24px;
}
.header h1 { font-size: 18px; color: #58a6ff; }
.header .stat { font-size: 13px; color: #8b949e; }
.header .stat b { color: #e0e0e0; }
.container { display: flex; height: calc(100vh - 52px); }
.left { flex: 1; border-right: 1px solid #30363d; display: flex; flex-direction: column; }
.right { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.panel-title {
  padding: 8px 16px; font-size: 13px; font-weight: 600;
  color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px;
  background: #161b22; border-bottom: 1px solid #30363d;
}
#graph { flex: 1; background: #0d1117; }
.flat-panel { flex: 1; overflow-y: auto; padding: 12px 16px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 6px 8px; color: #58a6ff; border-bottom: 1px solid #30363d; position: sticky; top: 0; background: #0f1117; }
td { padding: 5px 8px; border-bottom: 1px solid #21262d; vertical-align: top; }
tr:hover { background: #161b22; }
.obj-name { color: #7ee787; font-weight: 600; }
.pos { color: #d2a8ff; font-family: monospace; font-size: 12px; }
.step { color: #ffa657; }
.vqa-q { color: #79c0ff; }
.vqa-a { color: #e0e0e0; }
.vqa-item { padding: 6px 0; border-bottom: 1px solid #21262d; }
.vqa-step { color: #ffa657; font-size: 12px; }
.empty { color: #484f58; font-style: italic; padding: 20px; text-align: center; }
</style>
</head>
<body>
<div class="header">
  <h1>Spatial Memory</h1>
  <div class="stat">Explored: <b id="explored">0</b> viewpoints</div>
  <div class="stat">Objects: <b id="obj-count">0</b></div>
  <div class="stat">Relations: <b id="rel-count">0</b></div>
  <div class="stat" id="status" style="margin-left:auto; color:#3fb950;">● Live</div>
</div>
<div class="container">
  <div class="left">
    <div class="panel-title">Scene Graph</div>
    <div id="graph"></div>
  </div>
  <div class="right">
    <div class="panel-title">Flat Buffer</div>
    <div class="flat-panel" id="flat"></div>
    <div class="panel-title">VQA Results</div>
    <div class="flat-panel" id="vqa"></div>
  </div>
</div>
<script>
const EDGE_COLORS = {
  near: '#58a6ff', above: '#3fb950', below: '#f85149',
  left: '#ffa657', right: '#d2a8ff', in_front: '#79c0ff', behind: '#a371f7'
};

let network = null;

function buildGraph(data) {
  const nodes = [], edges = [];
  const objs = data.detected_objects || {};
  for (const [name, info] of Object.entries(objs)) {
    const hasPos = info.position !== null;
    nodes.push({
      id: name, label: name,
      color: hasPos ? { background: '#1f6feb', border: '#58a6ff' } : { background: '#21262d', border: '#484f58' },
      font: { color: '#e0e0e0', size: 14 },
      shape: hasPos ? 'dot' : 'diamond',
      size: 20,
      title: `step ${info.step ?? '?'}\npos: ${info.position ?? 'N/A'}\nsize: ${info.size ?? 'N/A'}`
    });
  }
  for (const rel of (data.relations || [])) {
    edges.push({
      from: rel.from, to: rel.to,
      label: rel.type, title: `${rel.from} → ${rel.to}: ${rel.type}`,
      color: { color: EDGE_COLORS[rel.type] || '#484f58', highlight: EDGE_COLORS[rel.type] || '#484f58' },
      font: { color: '#8b949e', size: 10 }, width: 1.5, smooth: { type: 'curvedCW', roundness: 0.15 }
    });
  }
  return { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
}

function renderFlat(data) {
  const objs = data.detected_objects || {};
  if (!Object.keys(objs).length) return '<div class="empty">No objects detected yet</div>';
  let html = '<table><tr><th>Object</th><th>Position</th><th>Size</th><th>Step</th><th>Crop</th></tr>';
  for (const [name, info] of Object.entries(objs)) {
    const pos = info.position ? `[${info.position.map(v=>v.toFixed(2)).join(', ')}]` : '—';
    const sz = info.size ? `[${info.size.map(v=>v.toFixed(2)).join(', ')}]` : '—';
    const crop = (info.crop_paths && info.crop_paths.length) ? info.crop_paths[0] : '—';
    html += `<tr><td class="obj-name">${name}</td><td class="pos">${pos}</td><td class="pos">${sz}</td><td class="step">${info.step ?? '—'}</td><td style="font-size:11px;max-width:200px;word-break:break-all">${crop}</td></tr>`;
  }
  return html + '</table>';
}

function renderVQA(data) {
  const vqas = data.vqa_results || [];
  if (!vqas.length) return '<div class="empty">No VQA results yet</div>';
  let html = '';
  for (const v of vqas) {
    html += `<div class="vqa-item"><span class="vqa-step">step ${v.step}</span><br><span class="vqa-q">Q: ${v.question}</span><br><span class="vqa-a">A: ${v.answer}</span></div>`;
  }
  return html;
}

function update(data) {
  document.getElementById('explored').textContent = data.explored_steps || 0;
  document.getElementById('obj-count').textContent = Object.keys(data.detected_objects || {}).length;
  document.getElementById('rel-count').textContent = (data.relations || []).length;
  document.getElementById('flat').innerHTML = renderFlat(data);
  document.getElementById('vqa').innerHTML = renderVQA(data);
  const gd = buildGraph(data);
  if (!network) {
    network = new vis.Network(document.getElementById('graph'), gd, {
      physics: { barnesHut: { gravitationalConstant: -3000, springLength: 150, springConstant: 0.02 } },
      interaction: { hover: true, tooltipDelay: 100 }
    });
  } else {
    network.setData(gd);
  }
}

async function poll() {
  try {
    const r = await fetch('/api/spatial-memory');
    if (r.ok) { update(await r.json()); document.getElementById('status').innerHTML = '● Live'; }
    else { document.getElementById('status').innerHTML = '● Error'; }
  } catch { document.getElementById('status').innerHTML = '● Disconnected'; }
}

poll();
setInterval(poll, 1000);
</script>
</body>
</html>"""


class SpatialMemoryVisualizer:
    """Flask-based web visualizer for SpatialMemory.

    Reads state from a JSON file and serves it via a web dashboard.
    Call `update(sm)` after each SpatialMemory update to refresh.
    """

    def __init__(
        self,
        json_path: str = DEFAULT_JSON_PATH,
        port: int = DEFAULT_PORT,
        host: str = "0.0.0.0",
    ) -> None:
        self.json_path = json_path
        self.port = port
        self.host = host
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

    def update(self, sm: Any) -> None:
        """Write SpatialMemory state to JSON file."""
        sm.save_json(self.json_path)

    def update_from_dict(self, data: dict) -> None:
        """Write raw dict to JSON file."""
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2)

    def start(self) -> None:
        """Start Flask server in a background thread."""
        if self._running:
            return

        from flask import Flask, jsonify, Response

        app = Flask(__name__)

        @app.route("/")
        def index():
            return Response(HTML_TEMPLATE, mimetype="text/html")

        @app.route("/api/spatial-memory")
        def api_spatial_memory():
            if os.path.exists(self.json_path):
                with open(self.json_path) as f:
                    return jsonify(json.load(f))
            return jsonify({"explored_steps": 0, "detected_objects": {}, "vqa_results": [], "relations": []})

        def run_server():
            app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

        self._running = True
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        print(f"[SpatialMemoryVisualizer] http://{self.host}:{self.port}")

    def stop(self) -> None:
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="Spatial Memory Web Visualizer")
    parser.add_argument("--json", default=DEFAULT_JSON_PATH, help="Path to spatial_memory.json")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--demo", action="store_true", help="Load demo data from training data")
    args = parser.parse_args()

    viz = SpatialMemoryVisualizer(json_path=args.json, port=args.port, host=args.host)

    if args.demo:
        from src.memory.spatial_memory import SpatialMemory
        sm = SpatialMemory()
        sm.update("GoNextPointTool", {}, "img1.jpg", 1)
        sm.update("GoNextPointTool", {}, "img2.jpg", 2)
        sm.update("ObjectLocation3D", {"object": "sofa"}, ([[2.1, 0.0, 1.5]], [[2.0, 1.5, 0.8]]), 3)
        sm.update("ObjectLocation3D", {"object": "curtain"}, ([[5.5, 4.3, 4.5]], [[0.2, 1.8, 0.5]]), 4)
        sm.update("ObjectLocation3D", {"object": "table"}, ([[2.5, 0.0, 1.2]], [[1.0, 0.5, 0.8]]), 5)
        sm.update("ObjectLocation2D", {"object": "sofa"}, {"bboxes_2d": [[10, 20, 110, 180]], "labels": ["sofa"]}, 3)
        sm.update("ObjectCrop", {"object": "sofa"}, ["cache/sofa_crop.jpg"], 3)
        sm.update("VisualQATool", {"question": "Is the sofa occupied?"}, "No, it is unoccupied.", 6)
        viz.update(sm)
        print("[demo] Loaded demo spatial memory data")

    viz.start()

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

/**
 * graph.js — D3.js force-directed knowledge graph visualization.
 * Adapted from graph_template.html.
 *
 * Public API:
 *   initGraph(graphData)  — render a graph in #graph-container
 *   clearGraph()          — clear the SVG
 */

const PALETTE = [
  "#4ECDC4", "#FF6B6B", "#60a5fa", "#FBBF24", "#86efac",
  "#c084fc", "#fb923c", "#f472b6", "#a78bfa", "#34d399",
  "#fca5a5", "#93c5fd", "#6ee7b7", "#fde68a", "#d8b4fe",
];

let _simulation = null;

function clearGraph() {
  if (_simulation) { _simulation.stop(); _simulation = null; }
  const root = document.getElementById("root");
  if (root) root.innerHTML = "";
  const legend = document.getElementById("legend");
  if (legend) legend.innerHTML = "";
  const detail = document.getElementById("detail-panel");
  if (detail) detail.classList.remove("visible");
  document.getElementById("stat-nodes").textContent = "—";
  document.getElementById("stat-edges").textContent = "—";
  document.getElementById("stat-communities").textContent = "—";
}

function initGraph(GRAPH_DATA) {
  clearGraph();

  if (!GRAPH_DATA || !GRAPH_DATA.nodes || GRAPH_DATA.nodes.length === 0) {
    document.getElementById("graph-empty").style.display = "flex";
    return;
  }
  document.getElementById("graph-empty").style.display = "none";

  // ── Dynamic COLOR_MAP ─────────────────────────────────────────────────────
  const uniqueTypes = [...new Set(GRAPH_DATA.nodes.map(n => n.type).filter(Boolean))];
  const COLOR_MAP = {};
  uniqueTypes.forEach((type, i) => {
    COLOR_MAP[type] = PALETTE[i % PALETTE.length];
  });
  COLOR_MAP["OTHER"] = COLOR_MAP["OTHER"] || "#94a3b8";

  function nodeColor(type) {
    return COLOR_MAP[type] || COLOR_MAP["OTHER"];
  }

  // ── Setup ─────────────────────────────────────────────────────────────────
  const svg       = d3.select("#graph");
  const root      = d3.select("#root");
  const tooltip   = document.getElementById("tooltip");
  const container = document.getElementById("graph-container");

  let W = container.clientWidth;
  let H = container.clientHeight;

  const zoom = d3.zoom()
    .scaleExtent([0.05, 6])
    .on("zoom", e => root.attr("transform", e.transform));
  svg.call(zoom);

  // ── Force simulation ──────────────────────────────────────────────────────
  const nodes = GRAPH_DATA.nodes.map(d => ({ ...d }));
  const links = GRAPH_DATA.links.map(d => ({ ...d }));

  const degreeMap = {};
  links.forEach(l => {
    degreeMap[l.source] = (degreeMap[l.source] || 0) + 1;
    degreeMap[l.target] = (degreeMap[l.target] || 0) + 1;
  });
  nodes.forEach(n => { n.degree = degreeMap[n.id] || 1; });

  const maxDeg = Math.max(...nodes.map(n => n.degree));
  const minDeg = Math.min(...nodes.map(n => n.degree));
  const nodeRadius = d => {
    const t = (d.degree - minDeg) / (maxDeg - minDeg || 1);
    return 8 + t * 28;
  };

  _simulation = d3.forceSimulation(nodes)
    .force("link",      d3.forceLink(links).id(d => d.id).distance(120))
    .force("charge",    d3.forceManyBody().strength(-300))
    .force("center",    d3.forceCenter(W / 2, H / 2))
    .force("collision", d3.forceCollide().radius(d => nodeRadius(d) + 6));

  // ── Draw links ────────────────────────────────────────────────────────────
  const linkGroup  = root.append("g").attr("class", "links");
  const labelGroup = root.append("g").attr("class", "edge-labels");
  const nodeGroup  = root.append("g").attr("class", "nodes");

  const link = linkGroup.selectAll("line")
    .data(links).join("line")
    .attr("class", "link")
    .attr("marker-end", "url(#arrow)");

  const edgeLabel = labelGroup.selectAll("text")
    .data(links).join("text")
    .attr("class", "edge-label")
    .text(d => d.label || "");

  // ── Draw nodes ────────────────────────────────────────────────────────────
  const node = nodeGroup.selectAll("g")
    .data(nodes).join("g")
    .attr("class", "node")
    .call(d3.drag()
      .on("start", dragStart)
      .on("drag",  dragged)
      .on("end",   dragEnd));

  node.append("circle")
    .attr("r", nodeRadius)
    .attr("fill", d => nodeColor(d.type));

  node.append("text")
    .attr("dy", d => nodeRadius(d) + 11)
    .text(d => d.label && d.label.length > 20 ? d.label.slice(0, 18) + "…" : (d.label || ""))
    .style("font-size", d => d.degree > 3 ? "11px" : "9px")
    .style("opacity", d => d.degree > 2 ? 1 : 0.5);

  // ── Simulation tick ───────────────────────────────────────────────────────
  _simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    edgeLabel
      .attr("x", d => (d.source.x + d.target.x) / 2)
      .attr("y", d => (d.source.y + d.target.y) / 2);
    node.attr("transform", d => `translate(${d.x},${d.y})`);
  });

  // ── Tooltip & selection ───────────────────────────────────────────────────
  let selectedNode = null;

  node
    .on("mouseover", (e, d) => {
      const connections = links.filter(l =>
        l.source.id === d.id || l.target.id === d.id).length;
      tooltip.innerHTML = `
        <div class="tt-name">${d.label}</div>
        <div class="tt-type" style="color:${nodeColor(d.type)}">${d.type}</div>
        ${d.description ? `<div class="tt-desc">${d.description}</div>` : ""}
        <div class="tt-connections"><span>${connections}</span> connection${connections !== 1 ? "s" : ""}</div>
      `;
      tooltip.classList.add("visible");
      positionTooltip(e);
    })
    .on("mousemove", positionTooltip)
    .on("mouseleave", () => tooltip.classList.remove("visible"))
    .on("click", (e, d) => {
      e.stopPropagation();
      if (selectedNode === d.id) {
        clearSelection();
        hideDetailPanel();
      } else {
        selectNode(d);
        showNodeDetail(d, links, nodes, nodeColor);
      }
    });

  // Edge click — show edge detail panel
  link.on("click", (e, d) => {
    e.stopPropagation();
    showEdgeDetail(d, nodeColor);
  });

  svg.on("click", () => { clearSelection(); hideDetailPanel(); });

  function positionTooltip(e) {
    const rect = container.getBoundingClientRect();
    let x = e.clientX - rect.left + 14;
    let y = e.clientY - rect.top  - 10;
    if (x + 280 > W) x = e.clientX - rect.left - 280;
    tooltip.style.left = x + "px";
    tooltip.style.top  = y + "px";
  }

  function selectNode(d) {
    selectedNode = d.id;
    const connectedIds = new Set([d.id]);
    links.forEach(l => {
      if (l.source.id === d.id) connectedIds.add(l.target.id);
      if (l.target.id === d.id) connectedIds.add(l.source.id);
    });
    node.classed("selected", n => n.id === d.id);
    node.classed("dimmed",   n => !connectedIds.has(n.id));
    link.classed("highlighted", l => l.source.id === d.id || l.target.id === d.id);
    link.classed("dimmed",      l => l.source.id !== d.id && l.target.id !== d.id);
  }

  function clearSelection() {
    selectedNode = null;
    node.classed("selected dimmed", false);
    link.classed("highlighted dimmed", false);
  }

  // ── Drag ──────────────────────────────────────────────────────────────────
  function dragStart(e, d) {
    if (!e.active) _simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
  }
  function dragged(e, d)  { d.fx = e.x; d.fy = e.y; }
  function dragEnd(e, d)  {
    if (!e.active) _simulation.alphaTarget(0);
    d.fx = null; d.fy = null;
  }

  // ── Legend ────────────────────────────────────────────────────────────────
  const typeCounts = {};
  nodes.forEach(n => { typeCounts[n.type] = (typeCounts[n.type] || 0) + 1; });

  const hiddenTypes = new Set();
  const legendEl = document.getElementById("legend");
  legendEl.innerHTML = "";

  Object.entries(COLOR_MAP).forEach(([type, color]) => {
    if (!typeCounts[type]) return;
    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = `
      <div class="legend-dot" style="background:${color}"></div>
      <span class="legend-name">${type}</span>
      <span class="legend-count">${typeCounts[type]}</span>
    `;
    item.addEventListener("click", () => {
      if (hiddenTypes.has(type)) {
        hiddenTypes.delete(type);
        item.classList.remove("disabled");
      } else {
        hiddenTypes.add(type);
        item.classList.add("disabled");
      }
      applyTypeFilter();
    });
    legendEl.appendChild(item);
  });

  function applyTypeFilter() {
    node.style("display", d => hiddenTypes.has(d.type) ? "none" : null);
    link.style("display", d => {
      const sHidden = hiddenTypes.has(d.source.type);
      const tHidden = hiddenTypes.has(d.target.type);
      return sHidden || tHidden ? "none" : null;
    });
  }

  // ── Search ────────────────────────────────────────────────────────────────
  const searchEl = document.getElementById("search");
  if (searchEl) {
    searchEl.value = "";
    searchEl.oninput = e => {
      const q = e.target.value.toLowerCase().trim();
      if (!q) { clearSelection(); hideDetailPanel(); return; }
      const match = nodes.find(n => n.label.toLowerCase().includes(q));
      if (match) { selectNode(match); showNodeDetail(match, links, nodes, nodeColor); }
    };
  }

  // ── Controls ──────────────────────────────────────────────────────────────
  const linkDistEl = document.getElementById("link-distance");
  if (linkDistEl) {
    linkDistEl.value = 120;
    linkDistEl.oninput = e => {
      _simulation.force("link").distance(+e.target.value);
      _simulation.alpha(0.3).restart();
    };
  }

  const chargeEl = document.getElementById("charge");
  if (chargeEl) {
    chargeEl.value = -300;
    chargeEl.oninput = e => {
      _simulation.force("charge").strength(+e.target.value);
      _simulation.alpha(0.3).restart();
    };
  }

  const labelThreshEl = document.getElementById("label-threshold");
  if (labelThreshEl) {
    labelThreshEl.value = 3;
    labelThreshEl.oninput = e => {
      const threshold = +e.target.value;
      edgeLabel.classed("visible", d => {
        const minD = Math.min(d.source.degree || 1, d.target.degree || 1);
        return minD >= threshold;
      });
    };
  }

  document.getElementById("btn-reset").onclick = () => {
    svg.transition().duration(500).call(
      zoom.transform, d3.zoomIdentity.translate(0, 0).scale(1)
    );
  };

  document.getElementById("btn-all").onclick = () => {
    hiddenTypes.clear();
    document.querySelectorAll(".legend-item").forEach(el => el.classList.remove("disabled"));
    applyTypeFilter();
    clearSelection();
    hideDetailPanel();
  };

  // ── Stats ─────────────────────────────────────────────────────────────────
  document.getElementById("stat-nodes").textContent = nodes.length;
  document.getElementById("stat-edges").textContent = links.length;
  document.getElementById("stat-communities").textContent =
    GRAPH_DATA.communities !== undefined ? GRAPH_DATA.communities : "—";

  // ── Resize ────────────────────────────────────────────────────────────────
  window.onresize = () => {
    W = container.clientWidth;
    H = container.clientHeight;
    _simulation.force("center", d3.forceCenter(W / 2, H / 2));
    _simulation.alpha(0.1).restart();
  };
}


// ── Detail panel (Obsidian-style) ─────────────────────────────────────────────

function showNodeDetail(d, links, nodes, nodeColor) {
  const panel = document.getElementById("detail-panel");
  const nodeIndex = {};
  nodes.forEach(n => { nodeIndex[n.id] = n; });

  const outgoing = links.filter(l => l.source.id === d.id);
  const incoming = links.filter(l => l.target.id === d.id);

  let relHTML = "";
  if (outgoing.length > 0) {
    relHTML += `<div class="dp-rel-header">Outgoing</div>`;
    outgoing.forEach(l => {
      const tgt = nodeIndex[l.target.id] || { label: l.target.id, type: "OTHER" };
      relHTML += `
        <div class="dp-rel-item" data-node-id="${tgt.id}" style="cursor:pointer;">
          <span class="dp-rel-label" style="color:${nodeColor ? nodeColor(tgt.type) : '#94a3b8'}">${l.label || "→"}</span>
          <span class="dp-rel-name">${tgt.label}</span>
          ${l.description ? `<div class="dp-rel-desc">${l.description}</div>` : ""}
        </div>`;
    });
  }
  if (incoming.length > 0) {
    relHTML += `<div class="dp-rel-header">Incoming</div>`;
    incoming.forEach(l => {
      const src = nodeIndex[l.source.id] || { label: l.source.id, type: "OTHER" };
      relHTML += `
        <div class="dp-rel-item" data-node-id="${src.id}" style="cursor:pointer;">
          <span class="dp-rel-label" style="color:${nodeColor ? nodeColor(src.type) : '#94a3b8'}">${l.label || "←"}</span>
          <span class="dp-rel-name">${src.label}</span>
          ${l.description ? `<div class="dp-rel-desc">${l.description}</div>` : ""}
        </div>`;
    });
  }

  panel.innerHTML = `
    <div class="dp-close" onclick="hideDetailPanel()">✕</div>
    <div class="dp-type" style="color:${nodeColor ? nodeColor(d.type) : '#94a3b8'}">${d.type}</div>
    <div class="dp-title">${d.label}</div>
    ${d.description ? `<div class="dp-description">${d.description}</div>` : ""}
    ${relHTML ? `<div class="dp-relations">${relHTML}</div>` : ""}
  `;

  panel.classList.add("visible");
}

function showEdgeDetail(d, nodeColor) {
  const panel = document.getElementById("detail-panel");
  const srcLabel = d.source.label || d.source.id || d.source;
  const tgtLabel = d.target.label || d.target.id || d.target;
  const srcType  = d.source.type || "";
  const tgtType  = d.target.type || "";

  panel.innerHTML = `
    <div class="dp-close" onclick="hideDetailPanel()">✕</div>
    <div class="dp-type">RELATIONSHIP</div>
    <div class="dp-title">${d.label || "connected to"}</div>
    ${d.description ? `<div class="dp-description">${d.description}</div>` : ""}
    <div class="dp-relations">
      <div class="dp-rel-header">Connects</div>
      <div class="dp-rel-item">
        <span class="dp-rel-label" style="color:${nodeColor ? nodeColor(srcType) : '#94a3b8'}">${srcType}</span>
        <span class="dp-rel-name">${srcLabel}</span>
      </div>
      <div class="dp-rel-item">
        <span class="dp-rel-label" style="color:${nodeColor ? nodeColor(tgtType) : '#94a3b8'}">${tgtType}</span>
        <span class="dp-rel-name">${tgtLabel}</span>
      </div>
    </div>
  `;

  panel.classList.add("visible");
}

function hideDetailPanel() {
  const panel = document.getElementById("detail-panel");
  if (panel) panel.classList.remove("visible");
}

import { useEffect, useRef } from "react";
import { DataSet, Network } from "vis-network/standalone";
import "vis-network/styles/vis-network.css";

export default function GraphVis({ nodes, edges }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current && nodes.length > 0) {

      const merged = simplifyEdges(edges);

      const data = {
        nodes: new DataSet(nodes),
        edges: new DataSet(merged)
      };

      const options = {
        nodes: {
          shape: "dot",
          size: 30,
          font: { size: 16 },
          borderWidth: 2
        },
        edges: {
          arrows: { to: { enabled: true, scaleFactor: 0.6 } },
          font: { align: "middle", size: 12 },
          smooth: {
            type: "dynamic",
            forceDirection: "horizontal",
            roundness: 0.4
          }
        },
        layout: {
          improvedLayout: true
        },
        physics: {
          enabled: true,
          barnesHut: {
            gravitationalConstant: -3000,
            springLength: 200,
            springConstant: 0.04,
            damping: 0.09
          }
        },
        interaction: {
          hover: true,
          tooltipDelay: 200,
          hideEdgesOnDrag: false
        }
      };

      new Network(containerRef.current, data, options);
    }
  }, [nodes, edges]);

  return (
    <div
      ref={containerRef}
      style={{ height: "700px", width: "100%", border: "1px solid #ddd", borderRadius: "8px", background: "#fff" }}
    />
  );
}

// here we merge edges that are bidirected
// so that it doesnt visually clutter the network graph
function simplifyEdges(edges) {
  const groups = edges.reduce((acc, { from, to, label, color }) => {
    const key = [from, to].sort().join('-')
    acc[key] = acc[key] || []
    acc[key].push({ from, to, label, color })
    return acc
  }, {})

  const dict = Object.values(groups).map((group, id) => {
    const { from, to, label, color } = group[0]
    const bidirectional = group.some(e => e.from === to && e.to === from)

    const arrows = bidirectional
      ? {
        to: { enabled: true, scaleFactor: 0.6 },
        from: { enabled: true, scaleFactor: 0.6 },
      }
      : {
        to: { enabled: true, scaleFactor: 0.6 }
      }

    return { id, from, to, label, color, arrows }
  })

  return dict
}


import { useEffect, useRef } from "react";
import { DataSet, Network } from "vis-network/standalone";
import "vis-network/styles/vis-network.css";

export default function GraphVis({ nodes, edges }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current && nodes.length > 0) {
      const data = {
        nodes: new DataSet(nodes),
        edges: new DataSet(edges)
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
            type: "cubicBezier",
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

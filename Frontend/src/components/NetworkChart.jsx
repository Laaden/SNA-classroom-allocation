import Plot from "react-plotly.js";

export default function NetworkChart() {
  // simple 2-point scatter to prove Plotly renders
  const data = [
    { x: [0, 1], y: [0, 1], mode: "markers+text", text: ["A", "B"] }
  ];

  const layout = {
    title: "Demo Network",
    autosize: true,
    margin: { t: 40, r: 20, b: 40, l: 20 }
  };

  return (
    <Plot
      data={data}
      layout={layout}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
      config={{ responsive: true }}
    />
  );
}

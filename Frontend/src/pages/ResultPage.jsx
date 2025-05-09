import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import GraphVis from "../components/GraphVis";
import "../styles/classforge.css";

const EDGE_COLORS = {
  Feedback: "#00BFFF",
  Advice: "#32CD32",
  Friends: "#FFA500",
  Disrespect: "#DC143C",
  "School Activities": "#9370DB",
  "More Time": "#FF69B4"
};

export default function ResultPage() {
  const [clusters, setClusters] = useState([]);
  const [selectedCluster, setSelectedCluster] = useState("");
  const [allNodes, setAllNodes] = useState([]);
  const [allEdges, setAllEdges] = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [studentList, setStudentList] = useState([]);
  const [edgeTypeFilter, setEdgeTypeFilter] = useState("All");

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [nodesRes, edgesRes] = await Promise.all([
          fetch("/api/result_node_cluster"),
          fetch("/api/result_edges_info")
        ]);

        if (!nodesRes.ok || !edgesRes.ok) {
          throw new Error("API fetch failed");
        }

        const nodesData = await nodesRes.json();
        const edgesData = await edgesRes.json();

        if (!Array.isArray(nodesData) || !Array.isArray(edgesData)) {
          throw new Error("API returned non-array data");
        }

        setAllNodes(nodesData);
        setAllEdges(edgesData);

        const foundClusters = [...new Set(nodesData.map(n => n.cluster_number))];
        setClusters(foundClusters);
        if (foundClusters.length > 0) {
          setSelectedCluster(foundClusters[0]);
        }
      } catch (err) {
        console.error("Failed to fetch API data:", err);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    if (!selectedCluster || !Array.isArray(allNodes) || !Array.isArray(allEdges)) return;

    const nodes = allNodes.filter(n => n.cluster_number === selectedCluster);
    const studentIds = new Set(nodes.map(n => n.id));

    const edges = allEdges
      .filter(e => studentIds.has(e.from)) // chỉ cần from là học sinh trong lớp
      .map(e => ({
        from: e.from.toString(),
        to: e.to.toString(),
        label: e.label || "",
        color: EDGE_COLORS[e.label] || "#ccc"
      }));

    const graphNodes = nodes.map(n => ({
      id: n.id.toString(),
      label: n.label
    }));

    setStudentList(nodes.map(n => n.label));
    setGraphData({ nodes: graphNodes, edges });
  }, [selectedCluster, allNodes, allEdges]);

  const handleClusterChange = (e) => {
    setSelectedCluster(Number(e.target.value));
    setEdgeTypeFilter("All");
  };

  const handleEdgeTypeChange = (e) => {
    const selected = e.target.value;
    setEdgeTypeFilter(selected);

    const clusterNodes = allNodes.filter(n => n.cluster_number === selectedCluster);
    const studentIds = new Set(clusterNodes.map(n => n.id));

    const filteredEdges = allEdges
      .filter(e => studentIds.has(e.from)) // vẫn chỉ lọc theo from
      .filter(e => selected === "All" || e.label === selected)
      .map(e => ({
        from: e.from.toString(),
        to: e.to.toString(),
        label: e.label || "",
        color: EDGE_COLORS[e.label] || "#ccc"
      }));

    setGraphData(prev => ({ ...prev, edges: filteredEdges }));
  };

  const edgeTypes = ["All", ...Array.from(new Set(allEdges.map(e => e.label)))];

  return (
    <div>
      <header>
        <div className="header-container">
          <div className="logo">
            <a href="index.html" className="logo-link">ClassForge</a>
          </div>
          <nav className="navbar">
            <ul className="nav-list">
              <li><Link to="/home">Home</Link></li>
              <li><Link to="/result">Dashboard</Link></li>
              <li><Link to="/weights">Allocation</Link></li>
              <li><a href="contact.html">Contact</a></li>
            </ul>
          </nav>
        </div>
      </header>

      <section id="dashboard">
        <div className="container">
          <h2>Dashboard</h2>
          <div className="dashboard-layout">
            <div className="dashboard-left">
              <div className="group-selection">
                <label htmlFor="clusterSelect"><strong>Select a Class:</strong></label>
                <select id="clusterSelect" value={selectedCluster} onChange={handleClusterChange}>
                  {clusters.map((cluster, index) => (
                    <option key={index} value={cluster}>{cluster}</option>
                  ))}
                </select>
              </div>

              <div className="student-list">
                <h3>Students in Class {selectedCluster}:</h3>
                <ul>
                  {studentList.map((student, index) => (
                    <li key={index}>{student}</li>
                  ))}
                </ul>
              </div>

              <div className="edge-filter">
                <label htmlFor="edgeFilter"><strong>Filter by Relationship Type:</strong></label>
                <select id="edgeFilter" value={edgeTypeFilter} onChange={handleEdgeTypeChange}>
                  {edgeTypes.map((type, index) => (
                    <option key={index} value={type}>{type}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="dashboard-right">
              <h3>Student Relationships in Class {selectedCluster}:</h3>
              {graphData.nodes.length > 0 ? (
                <GraphVis nodes={graphData.nodes} edges={graphData.edges} />
              ) : (
                <p>No data to display for this class.</p>
              )}
            </div>
          </div>
        </div>
      </section>

      <footer>
        <div className="container">
          <p>&copy; 2025 ClassForge. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

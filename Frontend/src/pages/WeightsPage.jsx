import { Link, useNavigate } from "react-router-dom";
import "../styles/classforge.css";
import React, { useState } from "react";

export default function WeightsPage() {
  const navigate = useNavigate();

  // --- academic weight (0–100) ---
  const [academicWeight, setAcademicWeight] = useState(50);

  // --- class-size weight (0–100) ---
  const [classSizeWeight, setClassSizeWeight] = useState(50);

  // --- SNA multipliers (–2.0 to +2.0) ---
  const [friendshipWeight, setFriendshipWeight] = useState(1.0);
  const [influenceWeight, setInfluenceWeight] = useState(1.0);
  const [feedbackWeight, setFeedbackWeight] = useState(1.0);
  const [adviceWeight, setAdviceWeight] = useState(1.0);
  const [disrespectWeight, setDisrespectWeight] = useState(1.0);
  const [affiliationWeight, setAffiliationWeight] = useState(1.0);

  // --- upload state ---
  const [datasetFile, setDatasetFile] = useState(null);
  const [collectionName, setCollectionName] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  // --- allocation feedback ---
  const [allocationMessage, setAllocationMessage] = useState("");

  // Handle file selection
  const handleFileChange = (e) => {
    if (e.target.files?.[0]) {
      setDatasetFile(e.target.files[0]);
    }
  };

  // Upload CSV to backend
  const handleDatasetUpload = async (e) => {
    e.preventDefault();
    if (!collectionName) {
      setUploadMessage("Please enter a collection name.");
      return;
    }
    if (!datasetFile) {
      setUploadMessage("Please select a dataset file.");
      return;
    }

    setIsUploading(true);
    setUploadMessage("Uploading...");
    const formData = new FormData();
    formData.append("file", datasetFile);

    try {
      const response = await fetch(
        `https://3.105.47.11:8000/upload_csv/${collectionName}`,
        { method: "POST", body: formData }
      );

      if (response.redirected) {
        window.location.href = response.url;
        return;
      }
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
      }
      const ct = response.headers.get("content-type") || "";
      if (ct.includes("application/json")) {
        const json = JSON.parse(await response.text());
        setUploadMessage(`✅ Upload successful: ${json.message || ""}`);
      } else {
        setUploadMessage("✅ Upload successful");
      }
    } catch (err) {
      console.error(err);
      setUploadMessage(
        err.message === "Failed to fetch"
          ? "❌ Error: Unable to connect to server."
          : `❌ Error: ${err.message}`
      );
    } finally {
      setIsUploading(false);
    }
  };

  // Run allocation
  const handleAllocation = async (e) => {
    e.preventDefault();
    const payload = {
      academic: academicWeight,
      classSize: classSizeWeight,
      friendship: friendshipWeight,
      influence: influenceWeight,
      feedback: feedbackWeight,
      advice: adviceWeight,
      disrespect: disrespectWeight,
      affiliation: affiliationWeight,
    };

    try {
      const response = await fetch("/api/allocate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(`Status ${response.status}`);
      const result = await response.json();
      setAllocationMessage(result.message || "Allocation successful.");
      navigate("/result");
    } catch (err) {
      console.error(err);
      setAllocationMessage(`Error running allocation: ${err.message}`);
    }
  };

  return (
    <div>
      {/* Header */}
      <header>
        <div className="header-container">
          <div className="logo">
            <a href="index.html" className="logo-link">
              ClassForge
            </a>
          </div>
          <nav className="navbar">
            <ul className="nav-list">
              <li><Link to="/home">Home</Link></li>
              <li><Link to="/result">Dashboard</Link></li>
              <li><Link to="/weights">Allocation</Link></li>
              <li><Link to="/ai-query">AI Query</Link></li>
            </ul>
          </nav>
        </div>
      </header>

      {/* Dataset Upload */}
      <section id="dataset-upload">
        <div className="container">
          <h2>Upload Dataset</h2>
          <form onSubmit={handleDatasetUpload}>
            <div className="form-group">
              <label htmlFor="collectionName">Collection Name:</label>
              <input
                type="text"
                id="collectionName"
                value={collectionName}
                onChange={e => setCollectionName(e.target.value)}
                placeholder="Enter collection name"
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="datasetFile">Select Dataset File (CSV):</label>
              <input
                type="file"
                id="datasetFile"
                accept=".csv"
                onChange={handleFileChange}
                required
              />
            </div>
            <button type="submit" disabled={isUploading}>
              {isUploading ? "Uploading..." : "Upload Dataset"}
            </button>
          </form>
          {uploadMessage && (
            <div className={uploadMessage.startsWith("❌") ? "error-message" : "success-message"}>
              {uploadMessage}
            </div>
          )}
        </div>
      </section>

      {/* Allocation */}
      <section id="allocation">
        <div className="container">
          <h2>Classroom Allocation</h2>
          <form id="allocationForm" onSubmit={handleAllocation}>

            {/* Academic 0–100 slider */}
            <div className="form-group">
              <label htmlFor="academicWeight">Academic Performance Weight:</label>
              <input
                type="range"
                id="academicWeight"
                min="0"
                max="100"
                value={academicWeight}
                onChange={e => setAcademicWeight(Number(e.target.value))}
              />
              <span>{academicWeight}%</span>
            </div>

            {/* Class-Size slider */}
            <div className="form-group">
              <label htmlFor="classSizeWeight">Class Size Weight:</label>
              <input
                type="range"
                id="classSizeWeight"
                min="0"
                max="100"
                value={classSizeWeight}
                onChange={e => setClassSizeWeight(Number(e.target.value))}
              />
              <span>{classSizeWeight}%</span>
            </div>

            {/* SNA multipliers –2.0 to +2.0 */}
            <div className="form-group">
              <label htmlFor="friendshipWeight">Friendship Multiplier:</label>
              <input
                type="range"
                id="friendshipWeight"
                min="-2"
                max="2"
                step="0.1"
                value={friendshipWeight}
                onChange={e => setFriendshipWeight(parseFloat(e.target.value))}
              />
              <span>{friendshipWeight.toFixed(1)}</span>
            </div>
            <div className="form-group">
              <label htmlFor="influenceWeight">Influence Multiplier:</label>
              <input
                type="range"
                id="influenceWeight"
                min="-2"
                max="2"
                step="0.1"
                value={influenceWeight}
                onChange={e => setInfluenceWeight(parseFloat(e.target.value))}
              />
              <span>{influenceWeight.toFixed(1)}</span>
            </div>
            <div className="form-group">
              <label htmlFor="feedbackWeight">Feedback Multiplier:</label>
              <input
                type="range"
                id="feedbackWeight"
                min="-2"
                max="2"
                step="0.1"
                value={feedbackWeight}
                onChange={e => setFeedbackWeight(parseFloat(e.target.value))}
              />
              <span>{feedbackWeight.toFixed(1)}</span>
            </div>
            <div className="form-group">
              <label htmlFor="adviceWeight">Advice Multiplier:</label>
              <input
                type="range"
                id="adviceWeight"
                min="-2"
                max="2"
                step="0.1"
                value={adviceWeight}
                onChange={e => setAdviceWeight(parseFloat(e.target.value))}
              />
              <span>{adviceWeight.toFixed(1)}</span>
            </div>
            <div className="form-group">
              <label htmlFor="disrespectWeight">Disrespect Multiplier:</label>
              <input
                type="range"
                id="disrespectWeight"
                min="-2"
                max="2"
                step="0.1"
                value={disrespectWeight}
                onChange={e => setDisrespectWeight(parseFloat(e.target.value))}
              />
              <span>{disrespectWeight.toFixed(1)}</span>
            </div>
            <div className="form-group">
              <label htmlFor="affiliationWeight">Affiliation Multiplier:</label>
              <input
                type="range"
                id="affiliationWeight"
                min="-2"
                max="2"
                step="0.1"
                value={affiliationWeight}
                onChange={e => setAffiliationWeight(parseFloat(e.target.value))}
              />
              <span>{affiliationWeight.toFixed(1)}</span>
            </div>

            <button type="submit">Run Allocation</button>
          </form>

          <div id="resultsContainer">
            {allocationMessage ? (
              <p>{allocationMessage}</p>
            ) : (
              <p>No results yet. Adjust preferences and run allocation.</p>
            )}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer>
        <div className="container">
          <p>&copy; 2025 ClassForge. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

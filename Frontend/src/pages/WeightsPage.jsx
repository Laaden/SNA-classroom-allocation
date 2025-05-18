import { Link, useNavigate } from "react-router-dom";
import "../styles/classforge.css";
import React, { useEffect, useState } from "react";

export default function WeightsPage() {
  const navigate = useNavigate();

  const [academicWeight, setAcademicWeight] = useState(50);
  const [friendshipWeight, setFriendshipWeight] = useState(1.0);
  const [influenceWeight, setInfluenceWeight] = useState(1.0);
  const [feedbackWeight, setFeedbackWeight] = useState(1.0);
  const [adviceWeight, setAdviceWeight] = useState(1.0);
  const [disrespectWeight, setDisrespectWeight] = useState(1.0);
  const [classSizeWeight, setClassSizeWeight] = useState(50);

  const [datasetFile, setDatasetFile] = useState(null);
  const [collectionName, setCollectionName] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [allocationMessage, setAllocationMessage] = useState("");

  const [isAllocating, setIsAllocating] = useState(false);

  useEffect(() => {
    async function getWeights() {
      try {
        const res = await fetch("http://3.105.47.11:8000/get_weights/");
        if (!res.ok) throw new Error(res.statusText);

        const data = await res.json();
        const setters = {
          academic: setAcademicWeight,
          classSize: setClassSizeWeight,
          friendship: setFriendshipWeight,
          influence: setInfluenceWeight,
          feedback: setFeedbackWeight,
          advice: setAdviceWeight,
          disrespect: setDisrespectWeight
        }

        for (const [k, v] of Object.entries(data)) {
          const setter = setters[k]
          if (setter) setter(v)
        }

      } catch (e) {
        console.error("Could not load weights:", e);
      }
    }
    getWeights()
  }, [])

  const handleFileChange = (e) => {
    if (e.target.files?.[0]) {
      setDatasetFile(e.target.files[0]);
    }
  };

  const handleDatasetUpload = async (e) => {
    console.log("🚨 handleDatasetUpload triggered");
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
        `http://3.105.47.11:8000/upload_csv/${collectionName}`,
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
        setUploadMessage(`? Upload successful: ${json.message || ""}`);
      } else {
        setUploadMessage("? Upload successful");
      }
    } catch (err) {
      console.error(err);
      setUploadMessage(
        err.message === "Failed to fetch"
          ? "? Error: Unable to connect to server."
          : `? Error: ${err.message}`
      );
    } finally {
      setIsUploading(false);
    }
  };

  const handleAllocation = async (e) => {
  e.preventDefault();
  setIsAllocating(true);
  try {
    const weightsPayload = {
      friendship: friendshipWeight,
      influence: influenceWeight,
      feedback: feedbackWeight,
      advice: adviceWeight,
      disrespect: disrespectWeight,
      classSize: classSizeWeight,
      academic: academicWeight
    };

    const weightsRes = await fetch("http://3.105.47.11:8000/update_weights/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(weightsPayload),
    });

    if (!weightsRes.ok) {
      throw new Error("Failed to update weights");
    }

    const runRes = await fetch("http://3.105.47.11:8000/run_gnn_all", {
      method: "POST",
    });

    const contentType = runRes.headers.get("content-type");

    if (!runRes.ok) {
      const errorText = await runRes.text();
      throw new Error(`GNN+GA execution failed: ${errorText}`);
    }

    let resultMessage = "Allocation completed successfully.";
    if (contentType && contentType.includes("application/json")) {
      const result = await runRes.json();
      resultMessage = result.message || resultMessage;
    }

    setAllocationMessage(resultMessage);
    navigate("/result");

  } catch (err) {
    console.error(err);
    setAllocationMessage(`? Error running allocation: ${err.message}`);
    setIsAllocating(false);
  }
};

  return (
    <div>
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
            <div className={uploadMessage.startsWith("?") ? "error-message" : "success-message"}>
              {uploadMessage}
            </div>
          )}
        </div>
      </section>

      <section id="allocation">
        <div className="container">
          <h2>Classroom Allocation</h2>
          <form id="allocationForm" onSubmit={handleAllocation}>
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
            <div className="form-group">
              <label htmlFor="classSizeWeight">Class Size Weight:</label>
              <input
                type="range"
                id="classSizeWeight"
                min="1"
                max="100"
                value={classSizeWeight}
                onChange={e => setClassSizeWeight(Number(e.target.value))}
              />
              <span>{classSizeWeight}</span>
            </div>
            <div className="form-group">
              <label htmlFor="friendshipWeight">Friendship Multiplier:</label>
              <input
                type="range"
                id="friendshipWeight"
                min="-10"
                max="10"
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
                min="-10"
                max="10"
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
                min="-10"
                max="10"
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
                min="-10"
                max="10"
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
                min="-10"
                max="10"
                step="0.1"
                value={disrespectWeight}
                onChange={e => setDisrespectWeight(parseFloat(e.target.value))}
              />
              <span>{disrespectWeight.toFixed(1)}</span>
            </div>
            <button type="submit" disabled={isAllocating}>
              {isAllocating ? "Running Allocation…" : "Run Allocation"}
            </button>

            {isAllocating && (
              <div className="loading-overlay">
                <div className="spinner" />
                <p>Crunching the numbers…</p>
              </div>
            )}

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

      <footer>
        <div className="container">
          <p>&copy; 2025 ClassForge. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
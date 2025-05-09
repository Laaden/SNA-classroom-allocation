import { Link, useNavigate } from "react-router-dom";
import "../styles/classforge.css";
import React, { useState } from "react";

export default function WeightsPage() {
    const navigate = useNavigate();
    const [academicWeight, setAcademicWeight] = useState(50);
    const [socialWeight, setSocialWeight] = useState(50);
    const [wellBeingWeight, setWellBeingWeight] = useState(50);
    const [allocationMessage, setAllocationMessage] = useState("");
    const [datasetFile, setDatasetFile] = useState(null);
    const [uploadMessage, setUploadMessage] = useState("");
    const [collectionName, setCollectionName] = useState("");
    const [isUploading, setIsUploading] = useState(false);
     
    // Handle file selection for dataset upload
    const handleFileChange = (e) => {
      if (e.target.files && e.target.files[0]) {
        setDatasetFile(e.target.files[0]);
      }
    };
  
    // Upload dataset file to the backend
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
        // Using the full URL instead of relative URL to avoid routing issues
        const response = await fetch(`/api/upload_csv/${collectionName}`, {
          method: "POST",
          body: formData
        });
        
        // Handle redirects as in the script
        if (response.redirected) {
          window.location.href = response.url;
          return;
        }
        
        // Check if response is OK
        if (!response.ok) {
          // Try to read as text first so we can see what's actually being returned
          const textContent = await response.text();
          
          // Try to parse as JSON if it looks like JSON
          let errorData;
          try {
            if (textContent && textContent.trim()) {
              errorData = JSON.parse(textContent);
            }
          } catch (jsonError) {
            // Not valid JSON, use text content as is
            throw new Error(`Upload failed: ${textContent || response.statusText}`);
          }
          
          throw new Error(`Upload failed: ${JSON.stringify(errorData) || response.statusText}`);
        }
        
        // Success case - first check if there is content
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
          // Only try to parse JSON if that's what we're expecting
          const text = await response.text();
          if (text.trim()) {
            const data = JSON.parse(text);
            setUploadMessage(`✅ Upload successful: ${data.message || "Dataset uploaded"}`);
          } else {
            setUploadMessage("✅ Upload successful");
          }
        } else {
          setUploadMessage("✅ Upload successful");
        }
      } catch (error) {
        console.error("Network error details:", error);
        if (error.message === "Failed to fetch") {
          setUploadMessage("❌ Error: Unable to connect to server. Please check if the server is running.");
        } else {
          setUploadMessage(`❌ Error: ${error.message}`);
        }
      } finally {
        setIsUploading(false);
      }
    };
  
    // Run the allocation algorithm with provided weights
    const handleAllocation = async (e) => {
      e.preventDefault();
      const payload = {
        academic: academicWeight,
        social: socialWeight,
        wellBeing: wellBeingWeight,
      };
  
      try {
        const response = await fetch("/api/allocate", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });
        
        if (!response.ok) {
          throw new Error(`Server responded with status: ${response.status}`);
        }
        
        const result = await response.json();
        setAllocationMessage(result.message || "Allocation successful and stored.");
        
        // Navigate to the result page after successful allocation
        navigate("/result");
      } catch (error) {
        console.error("Error running allocation:", error);
        setAllocationMessage(`Error running allocation: ${error.message}`);
      }
    };
  
    return (
      <div>
        {/* Header Section */}
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
  
        {/* Dataset Upload Section */}
        <section id="dataset-upload">
          <div className="container">
            <h2>Upload Dataset</h2>
            <form onSubmit={handleDatasetUpload}>
              <div className="form-group">
                <label htmlFor="collectionName">Collection Name:</label>
                <input
                  type="text"
                  id="collectionName"
                  name="collectionName"
                  value={collectionName}
                  onChange={(e) => setCollectionName(e.target.value)}
                  placeholder="Enter collection name"
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="datasetFile">Select Dataset File (CSV):</label>
                <input
                  type="file"
                  id="datasetFile"
                  name="datasetFile"
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
              <div className={uploadMessage.includes("Error") ? "error-message" : "success-message"}>
                {uploadMessage}
              </div>
            )}
          </div>
        </section>
  
        {/* Allocation Section */}
        <section id="allocation">
          <div className="container">
            <h2>Classroom Allocation</h2>
            <form id="allocationForm" onSubmit={handleAllocation}>
              <div className="form-group">
                <label htmlFor="academicWeight">Academic Performance Weight:</label>
                <input
                  type="range"
                  id="academicWeight"
                  name="academicWeight"
                  min="0"
                  max="100"
                  value={academicWeight}
                  onChange={(e) => setAcademicWeight(Number(e.target.value))}
                />
                <span>{academicWeight}%</span>
              </div>
              <div className="form-group">
                <label htmlFor="socialWeight">Social Integration Weight:</label>
                <input
                  type="range"
                  id="socialWeight"
                  name="socialWeight"
                  min="0"
                  max="100"
                  value={socialWeight}
                  onChange={(e) => setSocialWeight(Number(e.target.value))}
                />
                <span>{socialWeight}%</span>
              </div>
              <div className="form-group">
                <label htmlFor="wellBeingWeight">Well-Being Weight:</label>
                <input
                  type="range"
                  id="wellBeingWeight"
                  name="wellBeingWeight"
                  min="0"
                  max="100"
                  value={wellBeingWeight}
                  onChange={(e) => setWellBeingWeight(Number(e.target.value))}
                />
                <span>{wellBeingWeight}%</span>
              </div>
              <button type="submit">Run Allocation</button>
            </form>
            <div id="resultsContainer">
              {allocationMessage ? <p>{allocationMessage}</p> : <p>No results yet. Adjust preferences and run allocation.</p>}
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
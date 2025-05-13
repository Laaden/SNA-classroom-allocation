import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import "../styles/classforge.css";

export default function AIQueryPage() {
  const [userPrompt, setUserPrompt] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [apiEndpoint, setApiEndpoint] = useState(null);
  const [apiData, setApiData] = useState([]);
  const [isLoadingData, setIsLoadingData] = useState(false);
  const [error, setError] = useState(null);
  const [fields, setFields] = useState([]);
  const [collection, setCollection] = useState("");

  // Handle form submission to generate API endpoint
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!userPrompt.trim()) {
      setError("Please enter a prompt");
      return;
    }
    
    setIsSubmitting(true);
    setError(null);
    setApiEndpoint(null);
    setApiData([]);
    
    try {
      // Create form data for submission
      const formData = new FormData();
      formData.append("user_prompt", userPrompt);
      
      // Post to generate endpoint
      const response = await fetch("http://3.105.47.11:8000/generate", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      // Parse HTML response to extract the API endpoint URL
      // In a production app, you might want the backend to return JSON instead
      const htmlText = await response.text();
      
      // Extract URL from the HTML (simplified approach)
      const urlMatch = htmlText.match(/href=['"]([^'"]+)['"]/);
      if (urlMatch && urlMatch[1]) {
        const extractedUrl = urlMatch[1];
        setApiEndpoint(extractedUrl);
        
        // Extract fields (simplified approach)
        const fieldsMatch = htmlText.match(/Fields: ([^<]+)/);
        if (fieldsMatch && fieldsMatch[1]) {
          setFields(fieldsMatch[1].split(",").map(f => f.trim()));
        }
        
        // Extract collection name
        const collectionMatch = htmlText.match(/Collection: ([^<]+)/);
        if (collectionMatch && collectionMatch[1]) {
          setCollection(collectionMatch[1].trim());
        }
        
        // Now fetch data from the new endpoint
        await fetchDataFromEndpoint(extractedUrl);
      } else {
        throw new Error("Could not find API endpoint in the response");
      }
    } catch (err) {
      console.error("Error generating API:", err);
      setError(`Error: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Fetch data from the generated API endpoint
  const fetchDataFromEndpoint = async (endpoint) => {
    setIsLoadingData(true);
    try {
      const fullUrl = `http://3.105.47.11:8000${endpoint}`;
      const response = await fetch(fullUrl);
      
      if (!response.ok) {
        throw new Error(`API responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      setApiData(data);
    } catch (err) {
      console.error("Error fetching API data:", err);
      setError(`Error fetching data: ${err.message}`);
    } finally {
      setIsLoadingData(false);
    }
  };

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
              <li><Link to="/ai-query">AI Query</Link></li>
            </ul>
          </nav>
        </div>
      </header>

      <section id="ai-query">
        <div className="container">
          <h2>AI Query Generator</h2>
          
          <div className="query-form-container">
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="userPrompt">
                  Enter your query in natural language:
                </label>
                <textarea
                  id="userPrompt"
                  name="userPrompt"
                  value={userPrompt}
                  onChange={(e) => setUserPrompt(e.target.value)}
                  placeholder="Example: Show me all students with high attendance and low bullying scores"
                  rows={4}
                  required
                />
              </div>
              <button type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Generating API..." : "Generate API"}
              </button>
            </form>
            
            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
          </div>
          
          {apiEndpoint && (
            <div className="api-result-container">
              <div className="api-info">
                <h3>API Generated</h3>
                <p><strong>Endpoint:</strong> {apiEndpoint}</p>
                <p><strong>Collection:</strong> {collection}</p>
                <p><strong>Fields:</strong> {fields.join(', ')}</p>
              </div>
              
              <div className="api-data">
                <h3>Query Results</h3>
                {isLoadingData ? (
                  <p>Loading data...</p>
                ) : apiData.length > 0 ? (
                  <div className="data-table-container">
                    <table className="data-table">
                      <thead>
                        <tr>
                          {Object.keys(apiData[0]).map((key) => (
                            <th key={key}>{key}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {apiData.map((item, index) => (
                          <tr key={index}>
                            {Object.values(item).map((value, i) => (
                              <td key={i}>
                                {typeof value === 'object' 
                                  ? JSON.stringify(value) 
                                  : String(value)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p>No data found</p>
                )}
              </div>
            </div>
          )}
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
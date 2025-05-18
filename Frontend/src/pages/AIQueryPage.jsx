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
      const response = await fetch("http://3.105.47.11:8000/ask_agent", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({
          user_prompt: userPrompt,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      const result = await response.json();
      if (result.error) {
        throw new Error(result.error);
      }

      setApiEndpoint("/ask_agent");
      setApiData(result.results || []);
    } catch (err) {
      console.error("API Error:", err);
      setError(err.message);
    } finally {
      setIsSubmitting(false);
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

          <form onSubmit={handleSubmit} className="query-form-container">
            <div className="form-group">
              <label htmlFor="userPrompt">Enter your query in natural language:</label>
              <textarea
                id="userPrompt"
                name="userPrompt"
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                placeholder="Example: Show top 5 students with the most friends"
                rows={4}
                required
              />
            </div>
            <button type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Generating..." : "Make it work!"}
            </button>
          </form>

          {error && <div className="error-message">{error}</div>}

          {apiData.length > 0 && (
            <div className="api-data">
              <h3>Results</h3>
              <table className="data-table">
                <thead>
                  <tr>
                    {Object.keys(apiData[0]).map((key) => (
                      <th key={key}>{key}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {apiData.map((row, idx) => (
                    <tr key={idx}>
                      {Object.values(row).map((val, i) => (
                        <td key={i}>{typeof val === "object" ? JSON.stringify(val) : val}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
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

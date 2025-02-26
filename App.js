import React, { useState } from "react";
import axios from "axios";

function App() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [timeTaken, setTimeTaken] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponse("");
    setTimeTaken(null);

    try {
      const res = await axios.post("http://127.0.0.1:5000/query", { prompt });
      setResponse(res.data.response);
      setTimeTaken(res.data.time_taken);
    } catch (error) {
      setResponse("Error fetching response");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "50px", fontFamily: "Arial" }}>
      <h1>LLM Response Performance Tracker</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows="4"
          cols="50"
          placeholder="Enter your prompt..."
        />
        <br />
        <button type="submit" disabled={loading} style={{ marginTop: "10px" }}>
          {loading ? "Loading..." : "Submit"}
        </button>
      </form>

      {response && (
        <div style={{ marginTop: "20px" }}>
          <h2>Response:</h2>
          <p>{response}</p>
          <h3>Time Taken: {timeTaken ? timeTaken.toFixed(2) + "s" : "N/A"}</h3>
        </div>
      )}
    </div>
  );
}

export default App;

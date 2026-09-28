import { useEffect, useState } from "react";

function App() {
  const [backendStatus, setBackendStatus] = useState("Checking...");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/health")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Backend request failed");
        }
        return response.json();
      })
      .then((data) => {
        setBackendStatus(data.status === "ok" ? "Connected" : "Unavailable");
      })
      .catch(() => {
        setBackendStatus("Unavailable");
      });
  }, []);

  return (
    <div>
      <h1>Enterprise Knowledge & Action Agent</h1>

      <p>
        Backend Status: <strong>{backendStatus}</strong>
      </p>
    </div>
  );
}

export default App;
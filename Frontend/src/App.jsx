import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import HomePage     from "./pages/HomePage";
import ResultPage   from "./pages/ResultPage";
import WeightsPage  from "./pages/WeightsPage";
import AIQueryPage from './pages/AIQueryPage';

function App() {
  return (
    <Router basename="/SNA-classroom-allocation">
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/home" element={<HomePage />} />
        <Route path="/result" element={<ResultPage />} />
        <Route path="/weights" element={<WeightsPage />} />
        <Route path="/ai-query" element={<AIQueryPage />} /> {/* Add the new route */}
      </Routes>
    </Router>
  );
}

export default App;
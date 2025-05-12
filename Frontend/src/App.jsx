import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import HomePage     from "./pages/HomePage";
import ResultPage   from "./pages/ResultPage";
import WeightsPage  from "./pages/WeightsPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"          element={<HomePage />} />
        <Route path="/result"    element={<ResultPage />} />   {/* dashboard */}
        <Route path="/weights"   element={<WeightsPage />} />  {/* allocation */}
        <Route path="*"          element={<Navigate to="/" />} />
      </Routes>
    </BrowserRouter>
  );
}
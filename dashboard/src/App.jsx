import { Routes, Route, Navigate, useLocation } from "react-router-dom";
import SportNav from "./components/SportNav";
import NbaPage from "./pages/NbaPage";
import NflPage from "./pages/NflPage";
import AdminPage from "./pages/AdminPage";

// Multi-sport shell. Routes:
//   /       -> redirect to /nba
//   /nba    -> NBA predictor (existing behaviour, refactored)
//   /nfl    -> NFL predictor (new)
// A compact NBA/NFL switcher sits above the active page. Vercel rewrites all
// paths to index.html, so deep links and refreshes work for both routes.
export default function App() {
  const { pathname } = useLocation();
  const isAdmin = pathname.startsWith("/admin");
  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "1.5rem 1rem", fontFamily: "var(--font-sans, system-ui, sans-serif)", textAlign: "left" }}>
      {!isAdmin && (
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "1rem" }}>
          <SportNav />
        </div>
      )}
      <Routes>
        <Route path="/" element={<Navigate to="/nba" replace />} />
        <Route path="/nba" element={<NbaPage />} />
        <Route path="/nfl" element={<NflPage />} />
        <Route path="/admin" element={<AdminPage />} />
        <Route path="*" element={<Navigate to="/nba" replace />} />
      </Routes>
    </div>
  );
}

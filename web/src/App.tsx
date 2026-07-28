import { BrowserRouter, Link, Navigate, Route, Routes } from "react-router-dom";
import Chat from "./pages/Chat";
import Admin from "./pages/Admin";
import { IconShield } from "./components/icons";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      {/* 상단바에 관리자 링크를 두지 않는다.
          담당자 화면은 내부망에서 /admin 으로 직접 접속하는 경로만 남긴다 —
          고객 화면에서 관리 기능의 존재 자체를 노출하지 않기 위한 것이며,
          서버 쪽 ADMIN_ENABLED(기본 false)·토큰·IP 허용목록과 같은 방향의 조치다. */}
      <header className="topbar">
        {/* 브랜드가 상담 화면으로 돌아가는 유일한 이동 경로다 */}
        <Link to="/chat" className="brand"><IconShield size={20} /> KB 금융 소비자 보호 에이전트</Link>
      </header>
      <Routes>
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/admin" element={<Admin />} />
      </Routes>
    </BrowserRouter>
  );
}

import { Outlet } from "react-router-dom";
import Navbar from "@/components/navigation/Navbar";
import TopNavbar from "@/components/navigation/TopNavbar";

export default function DashboardLayout() {
  return (
    <div className="flex h-screen bg-linear-to-br from-slate-50 to-slate-100">
      <Navbar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <TopNavbar />
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

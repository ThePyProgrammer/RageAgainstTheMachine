import { useState } from "react";
import { Menu, X } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { routes } from "@/config/routes";
import { Button } from "@/components/ui/button";

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(true);
  const location = useLocation();

  const toggleNavbar = () => {
    setIsOpen(!isOpen);
  };

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <nav
      className={`bg-linear-to-b from-slate-900 to-slate-800 text-white transition-all duration-300 ease-in-out ${
        isOpen ? "w-64" : "w-20"
      } flex flex-col border-r border-slate-700 h-screen shadow-lg`}
    >
      {/* Header */}
      <div
        className={`flex items-center ${isOpen ? "justify-between" : "justify-center"} p-4 border-b border-slate-700/50`}
      >
        {isOpen && (
          <Link
            to="/"
            className="flex items-center gap-2 hover:opacity-80 transition-opacity"
          >
            <span className="text-lg font-bold text-blue-400">Thonk</span>
          </Link>
        )}
        <Button
          onClick={toggleNavbar}
          variant="ghost"
          size="icon"
          className="text-slate-300 hover:text-white cursor-pointer hover:bg-slate-700/50"
          aria-label="Toggle sidebar"
        >
          {isOpen ? <X size={20} /> : <Menu size={20} />}
        </Button>
      </div>

      {/* Navigation Items */}
      <div
        className={`flex-1 py-6 ${isOpen ? "px-3" : "px-2"} space-y-2 flex flex-col ${!isOpen && "items-center"}`}
      >
        {routes
          .filter((r) => r.nav)
          .map(({ path, label, icon: Icon }) => (
            <Link
              key={path}
              to={path}
              className={`flex items-center ${!isOpen && "justify-center"} gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                isActive(path)
                  ? "bg-linear-to-r from-blue-600 to-blue-500 text-white shadow-lg"
                  : "text-slate-300 hover:bg-slate-700/40 hover:text-white"
              } ${!isOpen && "w-12 h-12 p-0"}`}
              title={!isOpen ? label : ""}
            >
              {Icon ? <Icon size={20} className="shrink-0" /> : null}
              {isOpen && <span className="font-medium">{label}</span>}
            </Link>
          ))}
      </div>

      {/* Footer */}
      {/* <div className="p-4 border-t border-slate-700/50">
                {isOpen && (
                    <div className="text-xs text-slate-500">
                        <p>v1.0.0</p>
                    </div>
                )}
            </div> */}
    </nav>
  );
}

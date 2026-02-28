import { useLocation, Link } from "react-router-dom";
import { ChevronRight, Play, Square } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useBCIStream } from "@/hooks/useBCIStream";
import { useSessionStatus } from "@/contexts/SessionStatusContext";
import { routes } from "@/config/routes";

const breadcrumbMap: Record<
  string,
  Array<{ name: string; path: string }>
> = Object.fromEntries(
  routes.map((r) => [
    r.path,
    r.breadcrumbs ?? [{ name: r.label, path: r.path }],
  ]),
);

export default function TopNavbar() {
  const location = useLocation();
  const breadcrumbs = breadcrumbMap[location.pathname] || [
    { name: "Dashboard", path: "/" },
  ];
  const { status, executeStreamAction, isStreaming } = useBCIStream();
  const { status: sessionStatus } = useSessionStatus();

  return (
    <div className="bg-white border-b border-slate-200 shadow-sm h-15 flex items-center px-8 justify-between">
      <nav className="flex items-center gap-1" aria-label="Breadcrumb">
        {breadcrumbs.map((breadcrumb, index) => (
          <div key={breadcrumb.path} className="flex items-center gap-1">
            {index > 0 && (
              <ChevronRight size={18} className="text-slate-400 mx-1" />
            )}
            {index === breadcrumbs.length - 1 ? (
              <span className="text-slate-900 underline">
                {breadcrumb.name}
              </span>
            ) : (
              <Link
                to={breadcrumb.path}
                className="text-slate-600 hover:text-slate-900 transition-colors"
              >
                {breadcrumb.name}
              </Link>
            )}
          </div>
        ))}
      </nav>

      <div className="flex items-center gap-3">
        <Badge
          variant="outline"
          className={`text-xs px-2 py-0.5 font-medium border ${
            sessionStatus.eegStreaming
              ? "border-green-500 text-green-700 bg-green-50"
              : "border-slate-200 text-slate-500 bg-slate-50"
          }`}
        >
          EEG {sessionStatus.eegStreaming ? "Active" : "Inactive"}
        </Badge>

        <Badge
          variant="outline"
          className={`text-xs px-2 py-0.5 font-medium border ${
            sessionStatus.bfmProcessor
              ? "border-green-500 text-green-700 bg-green-50"
              : "border-slate-200 text-slate-500 bg-slate-50"
          }`}
        >
          BFM {sessionStatus.bfmProcessor ? "Running" : "Stopped"}
        </Badge>

        <Badge
          variant="outline"
          className={`text-xs px-2 py-0.5 font-medium border ${
            sessionStatus.webcam
              ? "border-green-500 text-green-700 bg-green-50"
              : "border-slate-200 text-slate-500 bg-slate-50"
          }`}
        >
          {sessionStatus.webcam
            ? `Webcam Active (Face ${sessionStatus.faceLocked ? "Locked" : "Unlocked"})`
            : "Webcam Inactive"}
        </Badge>

        <Badge
          variant="outline"
          className={`text-xs px-2 py-0.5 font-medium capitalize border ${
            status === "connected"
              ? "border-green-500 text-green-700 bg-green-50"
              : "border-slate-200 text-slate-500 bg-slate-50"
          }`}
        >
          Websocket {status}
        </Badge>

        <Button
          size="lg"
          className={`w-36 font-bold shadow-sm transition-all ${
            isStreaming
              ? "bg-red-50 text-red-600 border-2 border-red-200 hover:bg-red-100 cursor-pointer hover:border-red-300"
              : "bg-zinc-800 text-white hover:bg-zinc-500"
          }`}
          onClick={() => executeStreamAction(isStreaming ? "stop" : "start")}
        >
          {isStreaming ? (
            <>
              <Square size={16} fill="currentColor" className="mr-2" /> STOP
            </>
          ) : (
            <>
              <Play size={16} fill="currentColor" className="mr-2" /> STREAM
            </>
          )}
        </Button>
      </div>
    </div>
  );
}

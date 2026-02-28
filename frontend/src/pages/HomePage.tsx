import { Link } from "react-router-dom";
import {
  Brain,
  Heart,
  Activity,
  ArrowRight,
  BrainCog,
  Gauge,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function HomePage() {
  const sections = [
    {
      title: "EEG Streaming",
      description: "Real-time EEG signal monitoring and analysis",
      icon: Activity,
      color: "blue",
      path: "/eeg",
    },
    {
      title: "Command Centre",
      description: "Realtime focus, stress, flow and frustration signals",
      icon: Gauge,
      color: "teal",
      path: "/eeg/command-centre",
    },
    {
      title: "BFM Processor",
      description: "Brain Foundation Model embeddings",
      icon: Brain,
      color: "indigo",
      path: "/bfm",
    },
    {
      title: "Vitals Detector",
      description: "Webcam-based heart rate and pupillometry",
      icon: Heart,
      color: "red",
      path: "/webcam",
    },
    {
      title: "Motor Imagery",
      description: "Motor imagery classification and control",
      icon: BrainCog,
      color: "purple",
      path: "/mi",
    },
  ];

  const getColorClasses = (color: string) => {
    const colors: Record<string, { bg: string; hover: string; icon: string }> = {
      blue: {
        bg: "from-blue-500 to-blue-600",
        hover: "hover:bg-blue-600",
        icon: "bg-blue-500",
      },
      indigo: {
        bg: "from-indigo-500 to-indigo-600",
        hover: "hover:bg-indigo-600",
        icon: "bg-indigo-500",
      },
      teal: {
        bg: "from-teal-500 to-teal-600",
        hover: "hover:bg-teal-600",
        icon: "bg-teal-500",
      },
      red: {
        bg: "from-red-500 to-red-600",
        hover: "hover:bg-red-600",
        icon: "bg-red-500",
      },
      purple: {
        bg: "from-purple-500 to-purple-600",
        hover: "hover:bg-purple-600",
        icon: "bg-purple-500",
      },
    };
    return colors[color];
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-linear-to-br from-slate-50 to-slate-100 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-slate-900 mb-4">
            Thonk
          </h1>
          <p className="text-lg text-slate-600">
            Select a module to begin monitoring and analysis
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-5">
          {sections.map((section) => {
            const Icon = section.icon;
            const colors = getColorClasses(section.color);

            return (
              <Card
                key={section.path}
                className="border-slate-200 bg-white shadow-sm hover:shadow-lg transition-all hover:-translate-y-1"
              >
                <CardHeader className="pb-4">
                  <div className="flex items-center gap-3 mb-4">
                    <div
                      className={`bg-linear-to-br ${colors.bg} p-4 rounded-xl shadow-md`}
                    >
                      <Icon size={28} className="text-white" />
                    </div>
                  </div>
                  <CardTitle className="text-xl font-semibold text-slate-900">
                    {section.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-slate-600 mb-6 min-h-10">
                    {section.description}
                  </p>
                  <Link to={section.path}>
                    <Button
                      className={`w-full ${colors.icon} ${colors.hover} text-white group shadow-sm`}
                    >
                      Open Module
                      <ArrowRight
                        size={16}
                        className="ml-2 group-hover:translate-x-1 transition-transform"
                      />
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
}

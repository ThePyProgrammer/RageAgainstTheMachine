import HomePage from "@/pages/HomePage";
import BCIDashboardPage from "@/pages/EEGStreamingPage";
import BFMPage from "@/pages/BrainFoundationModelPage";
import PulseDetectorPage from "@/pages/WebcamPage";
import MotorImageryPage from "@/pages/MotorImageryPage";
import {
  Activity,
  Heart,
  Brain,
  BrainCog,
} from "lucide-react";
import type { ComponentType } from "react";

export type AppRoute = {
  path: string;
  label: string;
  element: ComponentType<Record<string, never>>;
  icon?: ComponentType<{ size?: number; className?: string }>;
  nav?: boolean; // include in sidebar nav
  breadcrumbs?: Array<{ name: string; path: string }>;
};

export const routes: AppRoute[] = [
  {
    path: "/",
    label: "Home",
    element: HomePage,
    nav: false,
    breadcrumbs: [{ name: "Home", path: "/" }],
  },
  {
    path: "/eeg",
    label: "EEG Streaming",
    element: BCIDashboardPage,
    icon: Activity,
    nav: true,
    breadcrumbs: [
      { name: "Home", path: "/" },
      { name: "EEG Streaming", path: "/eeg" },
    ],
  },
  {
    path: "/bfm",
    label: "BFM Processor",
    element: BFMPage,
    icon: Brain,
    nav: true,
    breadcrumbs: [
      { name: "Home", path: "/" },
      { name: "BFM Processor", path: "/bfm" },
    ],
  },
  {
    path: "/webcam",
    label: "Vitals Detector",
    element: PulseDetectorPage,
    icon: Heart,
    nav: true,
    breadcrumbs: [
      { name: "Home", path: "/" },
      { name: "Vitals Detector", path: "/webcam" },
    ],
  },
  {
    path: "/mi",
    label: "Motor Imagery",
    element: MotorImageryPage,
    icon: BrainCog,
    nav: true,
    breadcrumbs: [
      { name: "Home", path: "/" },
      { name: "Motor Imagery", path: "/mi" },
    ],
  },
];

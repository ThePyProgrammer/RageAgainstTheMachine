import HomePage from "@/pages/HomePage";
import BCIDashboardPage from "@/pages/EEGStreamingPage";
import BFMPage from "@/pages/BrainFoundationModelPage";
import PulseDetectorPage from "@/pages/WebcamPage";
import MotorImageryPage from "@/pages/MotorImageryPage";
import CommandCentreSignalsPage from "@/pages/CommandCentreSignalsPage";
import Combat3DPage from "@/pages/Combat3DPage";
import {
  Activity,
  Heart,
  Brain,
  BrainCog,
  Gauge,
  Gamepad2,
} from "lucide-react";
import type { ComponentType } from "react";
import PongPage from "@/pages/PongPage";
import BreakoutPage from "@/pages/BreakoutPage";
import FlappyBirdPage from "@/pages/FlappyBirdPage";

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
    path: "/eeg/command-centre",
    label: "Command Centre",
    element: CommandCentreSignalsPage,
    icon: Gauge,
    nav: true,
    breadcrumbs: [
      { name: "Home", path: "/" },
      { name: "Command Centre", path: "/eeg/command-centre" },
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
  {
    path: "/pong",
    label: "Pong",
    element: PongPage,
    icon: Gamepad2,
    nav: true,
    breadcrumbs: [{ name: "Home", path: "/" }, { name: "Pong", path: "/pong" }],
  },
  {
    path: "/breakout",
    label: "Breakout",
    element: BreakoutPage,
    icon: Gamepad2,
    nav: true,
    breadcrumbs: [{ name: "Home", path: "/" }, { name: "Breakout", path: "/breakout" }],
  },
  {
    path: "/combat3d",
    label: "Combat 3D",
    element: Combat3DPage,
    nav: true,
    breadcrumbs: [{ name: "Home", path: "/" }, { name: "Combat 3D", path: "/combat3d" }],
  },
  {
    path: "/flappy-bird",
    label: "Flappy Bird",
    element: FlappyBirdPage,
    icon: Gamepad2,
    nav: true,
    breadcrumbs: [{ name: "Home", path: "/" }, { name: "Flappy Bird", path: "/flappy-bird" }],
  },
];

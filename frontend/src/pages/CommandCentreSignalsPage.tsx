import { useEffect, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useCommandCentreSignals } from "@/hooks/useCommandCentreSignals";
import type { CommandCentreSignalKey } from "@/types/eeg";

type MetricDefinition = {
  key: CommandCentreSignalKey;
  label: string;
  description: string;
  formula: string;
  colorClass: string;
};

const METRICS: MetricDefinition[] = [
  {
    key: "focus",
    label: "Focus",
    description: "Frontal-midline theta vs baseline",
    formula: "frontal-midline-theta / baseline",
    colorClass: "bg-sky-500",
  },
  {
    key: "alertness",
    label: "Alertness",
    description: "Inverse frontal delta increase",
    formula: "1 - drowsiness",
    colorClass: "bg-emerald-500",
  },
  {
    key: "drowsiness",
    label: "Drowsiness",
    description: "Frontal delta increase vs baseline",
    formula: "frontal-delta / baseline",
    colorClass: "bg-amber-500",
  },
  {
    key: "stress",
    label: "Stress",
    description: "Higher beta-alpha ratio means higher stress",
    formula: "beta-alpha-ratio",
    colorClass: "bg-rose-500",
  },
  {
    key: "workload",
    label: "Workload",
    description: "Higher theta-alpha ratio means higher load",
    formula: "theta-alpha-ratio",
    colorClass: "bg-orange-500",
  },
  {
    key: "engagement",
    label: "Engagement",
    description: "Beta over alpha + theta",
    formula: "beta / (alpha + theta)",
    colorClass: "bg-indigo-500",
  },
  {
    key: "relaxation",
    label: "Relaxation",
    description: "Alpha + theta over beta",
    formula: "(alpha + theta) / beta",
    colorClass: "bg-teal-500",
  },
  {
    key: "flow",
    label: "Flow",
    description: "Engagement + moderate beta + low delta + balanced theta",
    formula: "composite index",
    colorClass: "bg-violet-500",
  },
  {
    key: "frustration",
    label: "Frustration",
    description: "Theta up + beta up + right-leaning theta asymmetry",
    formula: "composite index",
    colorClass: "bg-fuchsia-500",
  },
];

const formatPct = (value: number) => `${Math.round(value * 100)}%`;

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

function MetricCard({
  metric,
  value,
  previousValue,
}: {
  metric: MetricDefinition;
  value: number;
  previousValue: number;
}) {
  const delta = value - previousValue;
  const deltaLabel =
    delta > 0 ? `+${(delta * 100).toFixed(1)}%` : `${(delta * 100).toFixed(1)}%`;

  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-3">
          <CardTitle className="text-base text-slate-900">{metric.label}</CardTitle>
          <span className="text-lg font-semibold text-slate-900">
            {formatPct(value)}
          </span>
        </div>
        <p className="text-sm text-slate-600">{metric.description}</p>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="relative h-3 overflow-hidden rounded-full bg-slate-200">
          <div
            className={`h-full rounded-full transition-all duration-150 ${metric.colorClass}`}
            style={{ width: `${clamp01(value) * 100}%` }}
          />
          <div
            className="absolute inset-y-0 w-0.5 bg-slate-700/40"
            style={{ left: `${clamp01(previousValue) * 100}%` }}
          />
        </div>
        <div className="flex items-center justify-between text-xs">
          <span
            className={
              delta > 0
                ? "font-medium text-rose-600"
                : delta < 0
                  ? "font-medium text-emerald-600"
                  : "font-medium text-slate-500"
            }
          >
            {deltaLabel}
          </span>
          <span className="text-slate-500">{metric.formula}</span>
        </div>
      </CardContent>
    </Card>
  );
}

export default function CommandCentreSignalsPage() {
  const { signals, previousSignals, status, errorMessage, lastUpdated, deviceType } =
    useCommandCentreSignals();
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  const secondsSinceUpdate = lastUpdated
    ? Math.max(0, Math.floor((now - lastUpdated) / 1000))
    : null;

  return (
    <div className="min-h-full space-y-4 p-4">
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold text-slate-900">
            Command Centre Signals
          </h1>
          <Badge
            variant="outline"
            className={
              status === "connected"
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : "border-slate-200 bg-slate-50 text-slate-600"
            }
          >
            WS {status}
          </Badge>
          {deviceType ? (
            <Badge variant="outline" className="border-blue-200 bg-blue-50 text-blue-700">
              Device {deviceType}
            </Badge>
          ) : null}
          <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-600">
            Updated{" "}
            {secondsSinceUpdate === null ? "never" : `${secondsSinceUpdate}s ago`}
          </Badge>
        </div>
        <p className="mt-2 text-sm text-slate-600">
          Realtime cognitive proxies derived from EEG band-power ratios and
          baselines.
        </p>
      </div>

      {errorMessage ? (
        <Alert variant="destructive">
          <AlertTitle>Signal Stream Error</AlertTitle>
          <AlertDescription>{errorMessage}</AlertDescription>
        </Alert>
      ) : null}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
        {METRICS.map((metric) => (
          <MetricCard
            key={metric.key}
            metric={metric}
            value={signals[metric.key]}
            previousValue={previousSignals[metric.key]}
          />
        ))}
      </div>
    </div>
  );
}

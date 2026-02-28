import type { FunctionComponent } from "react";

type Step = "left" | "right" | "fine_tuning" | "complete" | "error";

type CalibrationWizardProps = {
  step: Step;
  instruction: string;
  progress: number;
  quality?: number;
  errorMessage?: string;
  running: boolean;
  onRetry?: () => void;
  onContinue?: () => void;
};

const TITLE_BY_STEP: Record<Step, string> = {
  left: "Look Left",
  right: "Look Right",
  fine_tuning: "Fine-Tuning Classifier",
  complete: "Calibration Complete",
  error: "Calibration Failed",
};

export const CalibrationWizard: FunctionComponent<CalibrationWizardProps> = ({
  step,
  instruction,
  progress,
  quality,
  errorMessage,
  running,
  onRetry,
  onContinue,
}) => {
  return (
    <section className="min-h-[calc(100vh-4rem)] bg-neutral-950 text-white p-6">
      <div className="mx-auto max-w-xl">
        <div className="h-2 bg-zinc-700 rounded">
          <div
            className="h-full bg-cyan-400 transition-all"
            style={{ width: `${Math.max(0, Math.min(1, progress)) * 100}%` }}
          />
        </div>

        <h2 className="mt-8 text-3xl font-bold tracking-tight">{TITLE_BY_STEP[step]}</h2>
        <p className="text-zinc-200 mt-3 text-lg">{instruction}</p>

        {running && (
          <p className="mt-4 text-sm text-cyan-300 animate-pulse">
            Capturing live EEG data...
          </p>
        )}

        {typeof quality === "number" && (
          <p className="mt-4 text-sm text-zinc-300">Best validation accuracy: {(quality * 100).toFixed(1)}%</p>
        )}

        {errorMessage && <p className="mt-4 text-sm text-rose-300">{errorMessage}</p>}

        <div className="mt-10 flex gap-3">
          {!running && onRetry && (
            <button
              type="button"
              className="rounded border border-yellow-500 px-4 py-2 hover:bg-yellow-500/10"
              onClick={onRetry}
            >
              Retry
            </button>
          )}
          {!running && onContinue && (
            <button type="button" className="rounded bg-cyan-500 px-4 py-2 text-black" onClick={onContinue}>
              Continue
            </button>
          )}
        </div>
      </div>
    </section>
  );
};

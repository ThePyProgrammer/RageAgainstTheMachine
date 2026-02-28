import type { FunctionComponent } from "react";

type Step = "baseline" | "left" | "right" | "complete";

type CalibrationWizardProps = {
  step: Step;
  trial: number;
  instruction: string;
  progress: number;
  quality?: number;
  onRetry?: () => void;
  onContinue: () => void;
};

export const CalibrationWizard: FunctionComponent<CalibrationWizardProps> = ({
  step,
  trial,
  instruction,
  progress,
  quality,
  onRetry,
  onContinue,
}) => {
  return (
    <section className="min-h-[calc(100vh-4rem)] bg-neutral-950 text-white p-6">
      <div className="mx-auto max-w-xl">
        <div className="h-2 bg-zinc-700 rounded">
          <div
            className="h-full bg-cyan-400 transition-all"
            style={{ width: `${progress * 100}%` }}
          />
        </div>

        <h2 className="mt-8 text-2xl font-bold">
          {step === "baseline" && "Baseline"}
          {step === "left" && `Left Trial ${trial}`}
          {step === "right" && `Right Trial ${trial}`}
          {step === "complete" && "Calibration Complete"}
        </h2>
        <p className="text-zinc-300 mt-2">{instruction}</p>

        {quality !== undefined && (
          <p className="mt-4 text-sm text-zinc-300">
            Quality {quality.toFixed(2)} {quality >= 1 ? "accepted" : "retry suggested"}
          </p>
        )}

        <p className="mt-12 text-xs text-zinc-500">
          This is a UI demo wizard. No hardware input is required.
        </p>

        <div className="mt-8 flex gap-3">
          {step === "complete" ? (
            <>
              {onRetry && (
                <button
                  type="button"
                  className="rounded border border-yellow-500 px-4 py-2"
                  onClick={onRetry}
                >
                  Retry
                </button>
              )}
              <button type="button" className="rounded bg-cyan-500 px-4 py-2" onClick={onContinue}>
                Continue
              </button>
            </>
          ) : (
            <button type="button" className="rounded bg-cyan-500 px-4 py-2" onClick={onContinue}>
              {step === "baseline" ? "Start" : "Next"}
            </button>
          )}
        </div>
      </div>
    </section>
  );
};

import { useState, useEffect } from "react";
import TimeseriesGraph from "@/components/eeg/TimeseriesGraph";
import { HeadPlot } from "@/components/eeg/HeadPlot3D";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { AlertCircle, X } from "lucide-react";
import { useBCIStream } from "@/hooks/useBCIStream";
import { useEmbeddings } from "@/hooks/useClassificationModelEmbeddings";
// import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

export default function BCIDashboardPage() {
  const { errorMessage, clearError, registerStopEmbeddings } = useBCIStream();
  const { disableEmbeddings } = useEmbeddings();
  const [isDismissed, setIsDismissed] = useState(false);

  // Register embedding stop callback
  useEffect(() => {
    registerStopEmbeddings(async () => {
      await disableEmbeddings();
    });
  }, [registerStopEmbeddings, disableEmbeddings]);

  const handleDismiss = () => {
    setIsDismissed(true);
    clearError();
  };

  return (
    <div className="h-full flex relative">
      {/* Central Error Popup */}
      {errorMessage && !isDismissed && (
        <div className="absolute inset-0 flex items-center justify-center z-50 pointer-events-none">
          <div className="max-w-lg w-full mx-4 pointer-events-auto">
            <Alert
              variant="destructive"
              className="shadow-2xl relative bg-white border-2 p-6"
            >
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 mt-0.5" />
                <div className="flex-1">
                  <AlertTitle className="text-lg mb-2">
                    Board Connection Error
                  </AlertTitle>
                  <AlertDescription className="text-base">
                    {errorMessage.includes("BOARD_NOT_READY") ||
                    errorMessage.includes("BOARD_NOT_CREATED")
                      ? "Unable to connect to the BCI board. Please ensure your board is powered on and properly connected via the dongle."
                      : errorMessage}
                  </AlertDescription>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleDismiss}
                  className="h-8 w-8 p-0 hover:bg-red-100 -mt-1 cursor-pointer"
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>
            </Alert>
          </div>
        </div>
      )}

      {/* Left Half*/}
      <div className="w-1/2 h-full">
        <TimeseriesGraph />
      </div>

      {/* Right Half */}
      <div className="w-1/2 h-full flex flex-col">
        <HeadPlot />

        {/* <Card className="flex-1 rounded-none border-0 flex flex-col">
                    <CardHeader className="pb-3 flex-none">
                        <CardTitle className="text-base">Notes</CardTitle>
                        <CardDescription className="text-xs">Session annotations</CardDescription>
                    </CardHeader>
                    <CardContent className="flex-1 pb-4 flex flex-col">
                        <textarea 
                            className="flex-1 w-full p-2 text-xs border rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-ring"
                            placeholder="Add notes about this session..."
                        />
                    </CardContent>
                </Card> */}
      </div>
    </div>
  );
}

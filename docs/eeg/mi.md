# Motor Imagery (MI) Processing

The Motor Imagery module translates EEG-captured brainwaves into actionable control commands. It takes streaming data, buffers it into specific windows, and processes it for inference based on foundations from [NeuralFlight](https://github.com/dronefreak/NeuralFlight).

## Implementation Details

The core processing logic is implemented in `backend/mi/services/mi_processor.py` and orchestrated through `mi_controller.py`.

### 1. Data Buffering and Epoching

- As continuous EEG data is streamed in, the `MIProcessor` collects samples using the active device's channel layout.
- For Muse v1, channels are reordered to `AF7, AF8, TP9, TP10` before buffering and inference.
- Epoch length is computed from model duration and live sampling rate (for Muse v1: ~768 samples for a 3-second window at 256 Hz).

### 2. Signal Resampling

- The extracted epoch is then resampled using `scipy.signal.resample`.
- Specifically, the live stream data is resampled from the hardware's native rate (e.g., 250 Hz) down to the rate the underlying machine learning model expects (default: 160 Hz, mapping 250 samples to 160 samples or 480 samples depending on the configuration).
- This ensures that the input dimensions exactly match the neural network's expected continuous input shape.

### 3. Classification Trigger

- Once an epoch is correctly shape-validated and resampled, a callback is fired.
- This invokes the loaded ML model prediction, sending the resulting probabilities and status updates back to the frontend dashboard over the WebSocket at `/mi/ws`.

### Dynamic Calibration

The backend also includes utilities (`calibration_manager.py` and `fine_tuner.py`) to manage dynamic model weights and fine-tune scaling parameters directly from the user's live session, helping to adapt the generic model to specific user baselines.

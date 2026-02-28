# Ocular Feature Extraction

The Ocular module analyzes the webcam video feed to extract critical metrics like pupil diameter, blink confidence, and eye detection.

## Implementation Details

The system leverages standard Haar Cascades paired with deep learning semantic segmentation via the [meye](https://github.com/fabiocarrara/meye) library.

### Processing Pipeline

1. **Face and Eye Detection (Haar Cascades)**:
   - The raw video frame is converted to grayscale.
   - `cv2.CascadeClassifier` applies `haarcascade_frontalface_alt.xml` to locate the user's face.
   - Within the face bounding box, `haarcascade_eye.xml` isolates the left and right eye regions.
   - The face is split vertically to definitively label the left and right eyes from the camera's perspective.

2. **Preprocessing for Neural Network**:
   - The extracted eye region patches are resized, grayscaled, and normalized into a tensor formatted for the deep learning model.

3. **Eye Segmentation via `meye`**:
   - The preprocessed tensor is passed to a pre-trained Keras `.h5` model (meye).
   - The model predicts two outputs: a pixel-level pupil segmentation map, and general tag confidences representing whether the image is actually an eye and whether a blink is occurring (`is_eye`, `is_blink`).

4. **Pupil Metric Calculation**:
   - A Non-Maximum Suppression (NMS) thresholding algorithm isolates the largest contiguous pupil region.
   - The center of mass and total pixel area of the pupil are calculated using `scipy.ndimage`.
   - **Physical Scaling**: Using an assumed average human iris diameter of roughly 11.7mm mapped against the bounding box size, the pixel area is mathematically converted into an estimated physical diameter in millimeters (`pupil_diameter_mm`).

These calculated metrics (diameter, coordinates, eye confidence, and blink presence) are continually streamed over the WebSocket.

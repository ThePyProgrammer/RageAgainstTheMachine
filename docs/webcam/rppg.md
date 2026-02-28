# rPPG Implementation Details

This document explains how vital signs from raw webcam video are extracted using the [rPPg-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox).

## 1. Pipeline

The system processes video frames in a continuous pipeline to extract three key metrics: **Heart Rate (BPM)**, **Oxygen Saturation (SpO2)**, and **Respiration Rate (RR)**.

How it works:

1.  **Face Detection**: Locks onto the user's face.
2.  **Signal Extraction (PhysNet)**: Uses a 3D-CNN to extract the raw Blood Volume Pulse (BVP) from video.
3.  **Signal Processing (DSP)**: Filters and cleans the signal.
4.  **Metric Calculation**: Derives BPM, SpO2, and RR from the clean signal.

## 2. Step-by-Step Breakdown

### Step 1: Face Detection & Locking

Locate the skin region of interest (ROI)

- **Detection**: Uses Haar Cascade classifier to find faces in the video stream.
- **Locking**: Once a face is found, the system "locks" onto it.
- **ROI Extraction**: Extracts the Forehead region, as it is vascular-rich and less affected by talking/mouth movement than the cheeks.

### Step 2: Signal Extraction

Convert video pixels into a 1D pulse wave (BVP).

Uses [**PhysNet**](https://bmvc2019.org/wp-content/uploads/papers/0186-paper.pdf), a state-of-the-art 3D Convolutional Neural Network.

- **Why PhysNet?**: Unlike traditional methods that just average the green channel, PhysNet learns spatial and temporal features. It can "see" the subtle skin color changes caused by blood flow while learning to ignore shadow changes and head motion.
- **Normalization**: Before entering the network, frames are normalized using Difference Normalization to remove the static skin tone and highlight temporal changes:
  $$ D(t) = \frac{C(t) - C(t-1)}{C(t) + C(t-1)} $$
- **Fallback**: If the model fails or is disabled, the system automatically falls back to the [**POS (Plane-Orthogonal-to-Skin)**](https://ieeexplore.ieee.org/document/7565547) algorithm, a mathematical method that projects RGB signals onto a plane orthogonal to the skin tone.

### Step 3: Post-Processing & Cleanup

Remove noise (lighting flicker, slow movements) from the raw BVP.

1.  **Detrending**: Applies detrending to remove the "DC drift" (slow changes in lighting or auto-exposure).
2.  **Bandpass Filtering**: Applies a filter to keep only physiologically possible frequencies.

### Step 4: Metric Calculation

#### A. Heart Rate (BPM)

1.  **FFT Method**: Converts the signal to the frequency domain to find the dominant periodic component.
2.  **Peak Detection**: Counts the actual peaks in the time-domain signal.
3.  **Fusion Logic**:
    - The system compares outcomes from both methods.
    - It checks Historical Stability (is this BPM close to the last 10s average?).
    - It uses a Signal-to-Noise Ratio (SNR) check to resolve disagreements (e.g., if FFT says 60 but Peak says 120, which signal is "cleaner"?).
    - This prevents the system from locking onto harmonics or noise.

#### B. Oxygen Saturation (SpO2)

Uses the Ratio of Ratios principle, analyzing the pulsatile (AC) vs static (DC) components of Red and Blue light.

$$ R = \frac{(AC*{red} / DC*{red})}{(AC*{blue} / DC*{blue})} $$

- **Refinement**: Applies Bandpass Filter to the Red/Blue signals before calculating the AC component (Standard Deviation). This ensures that noise (like shadow movement) doesn't get counted as "pulse amplitude," preventing artificially low readings.
- **Calibration**: The values are mapped to percentage using empirically derived coefficients ($SpO2 = A - B \times R$).

#### C. Respiration Rate (RR)

We calculate breathing rate using **RSA (Respiratory Sinus Arrhythmia)**.

- When you inhale, your heart rate speeds up slightly. When you exhale, it slows down.
- **Algorithm**:
  1.  Detect individual heartbeats (peaks) in the BVP signal.
  2.  Calculate the Inter-Beat Intervals (IBI) (time between beats).
  3.  Analyze the periodic fluctuation of these IBIs.
  4.  The frequency of this fluctuation is your Respiration Rate.

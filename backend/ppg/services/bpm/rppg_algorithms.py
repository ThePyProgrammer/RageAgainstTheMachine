import math
import numpy as np
from scipy import signal

from shared.config.logging import get_logger

logger = get_logger(__name__)


def POS(RGB, fs=30):
    """
    Plane-Orthogonal-to-Skin (POS) algorithm for BVP extraction.

    Args:
        RGB: (L, 3) numpy array of RGB signals (Red, Green, Blue)
        fs: Sampling frequency (default 30Hz)

    Returns:
        BVP: (L,) numpy array of Blood Volume Pulse signal
    """
    WinSec = 1.6
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            # Normalize temporal window
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            # Projection
            Cn = np.asmatrix(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)

            # Tuning
            std_s0 = np.std(S[0, :])
            std_s1 = np.std(S[1, :])

            # Prevent division by zero when signal has no variation
            if std_s1 > 1e-6:
                h = S[0, :] + (std_s0 / std_s1) * S[1, :]
            else:
                # Fallback: use only first component if second has no variation
                h = S[0, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h

            # Overlap-add
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H[0]  # Return flat array
    return BVP


def calculate_spo2(red_segment, blue_segment, fs=30.0):
    """
    Calculate SpO2 using Ratio of Ratios with filtered AC components.

    Args:
        red_segment: Numpy array of Red channel intensities
        blue_segment: Numpy array of Blue channel intensities (or Green)
        fs: Sampling frequency

    Returns:
        float: Estimated SpO2 percentage
    """
    if len(red_segment) == 0 or len(blue_segment) == 0:
        return 0.0

    # 1. Calculate DC components from RAW signal (Mean)
    dc_r = np.mean(red_segment)
    dc_b = np.mean(blue_segment)

    if dc_r == 0 or dc_b == 0:
        return 0.0

    # 2. Calculate AC components from FILTERED signal (Std Dev)
    # Bandpass filter [0.5 - 4.0 Hz] to remove DC drift and high freq noise
    nyquist = 0.5 * fs
    if nyquist <= 0:
        return 0.0

    low = 0.5 / nyquist
    high = 4.0 / nyquist

    # Clamp to valid range (0, 1)
    if high >= 1.0:
        high = 0.99
    if low <= 0.0:
        low = 0.01

    if low >= high or low >= 1.0:
        # Fallback if fs is weird
        ac_r = np.std(red_segment)
        ac_b = np.std(blue_segment)
    else:
        b, a = signal.butter(3, [low, high], btype="bandpass")

        # Filter Red
        filtered_r = signal.filtfilt(b, a, red_segment)
        ac_r = np.std(filtered_r)

        # Filter Blue
        filtered_b = signal.filtfilt(b, a, blue_segment)
        ac_b = np.std(filtered_b)

    # 3. Calculate Ratio of Ratios
    # R = (AC_red/DC_red) / (AC_blue/DC_blue)
    if ac_b == 0:
        return 0.0

    R = (ac_r / dc_r) / (ac_b / dc_b)

    # 4. Calibration
    # Empirically derived coefficients
    A = 110
    B = 15

    spo2 = A - B * R
    return max(0.0, min(100.0, spo2))


def calculate_rr(bvp_signal: np.ndarray, fs: float = 30.0) -> float:
    """
    Calculate Respiration Rate (RR) using RSA (Respiratory Sinus Arrhythmia).
    Analyzes the variability in peak-to-peak intervals of the pulse.

    Args:
        bvp_signal: Clean BVP signal
        fs: Sampling frequency

    Returns:
        float: Respiration Rate in breaths/min
    """
    # Need reasonable duration for RSA (at least 8-10s)
    if bvp_signal is None or len(bvp_signal) < int(8 * fs):
        return 0.0

    # 1. Find Peaks to get IBIs
    # Distance corresponding to max HR 200bpm = 0.3s
    distance = max(
        1, int(0.3 * fs)
    )  # enforce min dist of 1 -> if fs drops, distance must NEVER be 0

    try:
        peaks, _ = signal.find_peaks(bvp_signal, distance=distance)
        if len(peaks) < 4:
            return 0.0

        # 2. Calculate Inter-Beat Intervals (IBIs)
        # These are sample indices, convert to time
        peak_times = peaks / fs
        ibis = np.diff(peak_times)  # Time between beats in seconds

        # 3. Create IBI Time Series
        # The IBI series is unevenly sampled (at each beat).
        # Use the time of the *second* peak in each pair as the timestamp for that IBI.
        ibi_times = peak_times[1:]

        # 4. Resample to uniform grid for FFT/Filtering
        # We want to detect breathing (0.1 - 0.5 Hz).
        # Resample at 4Hz (plenty for 0.5Hz signal)
        resample_fs = 4.0
        duration = ibi_times[-1] - ibi_times[0]

        if duration < 5.0:  # Need some duration to see breathing
            return 0.0

        num_points = int(duration * resample_fs)
        uniform_times = np.linspace(ibi_times[0], ibi_times[-1], num_points)

        # Linear interpolation
        uniform_ibis = np.interp(uniform_times, ibi_times, ibis)

        # 5. Extract Breathing Signal from IBI variability
        # Detrend to remove mean HR changes (very low freq)
        uniform_ibis = signal.detrend(uniform_ibis)

        # 6. Find dominant frequency in breathing range [0.1 - 0.5 Hz] (6-30 bpm)
        # Using simple peak counting on the filtered IBI signal is often robust.

        nyquist = 0.5 * resample_fs
        low = 0.1 / nyquist
        high = 0.5 / nyquist
        b, a = signal.butter(2, [low, high], btype="bandpass")
        resp_signal = signal.filtfilt(b, a, uniform_ibis)

        # Count peaks in the RSA signal
        # Min distance for 30 breaths/min at 4Hz = ~0.13s (too small),
        # use 30bpm -> 0.5Hz -> 2.0s period -> peaks 2s apart? No, that's period.
        # 30 bpm = 0.5Hz. Period = 2s. Samples = 2 * 4 = 8 samples.
        resp_peaks, _ = signal.find_peaks(resp_signal, distance=8)

        num_breaths = len(resp_peaks)

        if num_breaths < 2:
            # Fallback: FFT method if peak counting fails
            return 0.0

        # Calculate rate based on duration of resampled signal
        rr = (num_breaths / duration) * 60.0
        return float(rr)

    except Exception as e:
        logger.error("Error in RR calculation: %s", e, exc_info=True)
        return 0.0

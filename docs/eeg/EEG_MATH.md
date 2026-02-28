# EEG Signal Processing Logic

Below details the mathematical logic used to determine signal quality and signal power.

## 1. The Hardware Limit

The ADS1299 chip (the brain of the Cyton board) has a physical input ceiling. If a signal exceeds this limit, the amplifier saturates ("clips"), making the data invalid.

* **$V_{REF}$ (Reference Voltage):** 4.5 Volts. The Cyton uses an internal 4.5V reference.
* **Gain:** 24x. The default magnification setting for EEG channels.

To find the maximum readable voltage ($V_{MAX}$), we divide the reference voltage by the gain:

$$\text{Limit} = \frac{4.5 \text{ Volts}}{24} = 0.1875 \text{ Volts}$$

Converting to Microvolts ($\mu V$):

$$0.1875 \times 1,000,000 = \mathbf{187,500 \mu V}$$

If a signal hits $+187,500$ or $-187,500$, it is physically impossible for the hardware to go higher. The signal is "Railed."

## 2. "Railed" Status (Connection Quality)

To determine connection quality, we measure how close the signal is floating toward the hardware limit.

We calculate the percentage of the limit used by the strongest signal peak in the current time window:

$$\text{Offset \%} = \left( \frac{|\text{Max Signal Value}|}{187,500} \right) \times 100$$

The official OpenBCI thresholds to categorize the signal status:

* **🟢 Normal:** **0% - 75%**
  * Signal is centered and valid.
* **🟡 Near Railed (Warning):** **> 75%**
  * The signal is drifting significantly. Usually indicates weak electrode contact or static buildup.
* **🔴 Railed (Error):** **> 90%**
  * The signal is invalid. The electrode is likely disconnected, floating, or touching another piece of metal.

## 3. RMS (Signal Power)

RMS (Root Mean Square) tells us how "loud" or active the brainwaves are.

We use the Standard Deviation method to calculate RMS.

### Why Standard Deviation?

Raw EEG signals often have a "DC Offset" (they float at 50,000 µV instead of 0 µV). If we calculated raw RMS, the result would be huge (50,000).

Standard Deviation mathematically removes the average drift (Mean) and only measures the fluctuations (AC Power) around that mean.

$$\text{RMS} = \sqrt{\frac{\sum(x_i - \mu)^2}{N}}$$

* $x_i$: Individual voltage sample.
* $\mu$: Mean (Average) voltage of the window.
* $N$: Total number of samples.

## Reference

* **Source Code:** [OpenBCI_GUI/Extras.pde (Lines 470-557)](https://github.com/OpenBCI/OpenBCI_GUI/blob/e23869e7b5cc621e733d8fa0d81f05d477264306/OpenBCI_GUI/Extras.pde#L480)
  
from datetime import datetime


class DataProcessor:
    @staticmethod
    def process_batch(data, channel_info, sample_counter):
        rows = []
        for i, col in enumerate(data.T):
            ts = col[channel_info["ts_idx"]]
            ts_ms = int(ts * 1000)
            ts_fmt = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
            marker = (
                col[channel_info["marker_idx"]]
                if channel_info["marker_idx"] is not None
                else 0
            )

            row = [sample_counter, ts_ms, ts_fmt, marker]
            row.extend(col[channel_info["eeg_channels"]].tolist())
            row.extend(col[channel_info["accel_channels"]].tolist())
            row.extend(col[channel_info["analog_channels"]].tolist())

            rows.append(row)
            sample_counter += 1
        return rows

    @staticmethod
    def build_header():
        base_header = [
            "sample_index",
            "ts_unix_ms",
            "ts_formatted",
            "marker",
            "eeg_ch_0",
            "eeg_ch_1",
            "eeg_ch_2",
            "eeg_ch_3",
            "eeg_ch_4",
            "eeg_ch_5",
            "eeg_ch_6",
            "eeg_ch_7",
            "accel_x",
            "accel_y",
            "accel_z",
            "analog_0",
            "analog_1",
            "analog_2",
        ]
        return base_header

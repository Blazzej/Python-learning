# TODO
# Check if singals lenght is above some seconds
# If signal lost for some time, reset count or finish detection

import pyaudio
import time
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import queue
import datetime

RATE = 44100
CHUNK = 4096
FORMAT = pyaudio.paFloat32
CHANNELS = 1
TIME_BUFFER = 5 # seconds
MIN_MAG = 0.05
MIN_SIGNAL_DURATION = 0.7 # seconds
SIGNAL_CHANGE_TIME = 0.5 # seconds
FREQ_TOL = 20 # Hz
MAX_SIGNAL_LENTH = 5 # seconds



def is_in_range(f,targets):
    for target_freq in targets:
                    if np.abs(f - target_freq) < 20:
                        return True
    return False

def find_device_index(p, name_substr):
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if name_substr.lower() in info.get('name', '').lower():
            return i
    return None

def parabolic_peak_fit(mags, index, xf, rate, chunk):

    if index <= 0 or index >= len(mags) - 1:
        return xf[index]
    
    y1, y2, y3 = mags[index-1], mags[index], mags[index+1]
    
    numerator = y1 - y3
    denominator = y1 - 2 * y2 + y3
    
    if denominator == 0:
        return xf[index]
    
    p = 0.5 * numerator / denominator
    
    delta_f = rate / chunk 
    
    return xf[index] + p * delta_f

def finish_detection(pyaudio_instance, signals_count, stream):
    print("Stopping stream...(function)")
    stream.stop_stream()
    stream.close()
    pyaudio_instance.terminate()
    print("Done.")
    if signals_count == 1:
        print("One signal detected.")
    else:
        print("Two signals detected.")

def audio_callback(in_data, frame_count, time_info, status):

    try:
        arr = np.frombuffer(in_data, dtype=np.float32)
    except Exception:
        return (None, pyaudio.paContinue)

    if len(arr) < CHUNK:
        arr = np.pad(arr, (0, CHUNK - len(arr)), 'constant')

    windowed = arr[:CHUNK] * np.hanning(CHUNK)
    yf = np.fft.rfft(windowed)
    xf = np.fft.rfftfreq(CHUNK, 1.0 / RATE)
    mags = np.abs(yf)
    max_mag = np.max(mags)

    peaks, _ = find_peaks(mags, height=np.max(mags)*0.1, distance=5)
    
    dom_freq = None
    if len(peaks) > 0:

        dom_idx = peaks[np.argmax(mags[peaks])]

        dom_freq = parabolic_peak_fit(mags, dom_idx, xf, RATE, CHUNK)
        
    try:
        result_queue.put_nowait((dom_freq, xf, mags, max_mag))
    except queue.Full:
        pass

    usable_data = np.frombuffer(in_data, dtype=np.float32).copy()
    usable_data *= 0.2
    out_data = usable_data.tobytes()
    return (out_data, pyaudio.paContinue) 

def main():
    
    signals_freqs = [600, 880, 1000, 1200]
    signal_start_time = signal_finish_time = datetime.datetime.now()
    signals_count = 0


    global result_queue
    result_queue = queue.Queue(maxsize=8)

    p = pyaudio.PyAudio()

    micro_idx = find_device_index(p, "main_micro") 
    speaker_idx = find_device_index(p, "jbl_go")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    frames_per_buffer=CHUNK,
                    input=True,
                    output=True,
                    input_device_index=micro_idx,
                    output_device_index=speaker_idx,
                    stream_callback=audio_callback)


    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlim(0, RATE/2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Hz")
    ax.set_ylabel("Magnitude")
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    stream.start_stream()
    print(f"Stream started on device: {p.get_device_info_by_index(micro_idx)['name']}")

    try:
        while stream.is_active():
            now = datetime.datetime.now()
            try:
                dom_freq, xf, mags, max_mag = result_queue.get(timeout=0.1)
            except queue.Empty:
                time.sleep(0.01)
                continue

            if dom_freq is not None and is_in_range(dom_freq, signals_freqs) and max_mag > MIN_MAG:
                #print(f"Dominant Frequency: {dom_freq:.3f} Hz")
            
                for target_freq in signals_freqs:
                    if np.abs(dom_freq - target_freq) < FREQ_TOL:
                        if (now - signal_finish_time).total_seconds() < SIGNAL_CHANGE_TIME:
                            if (signal_finish_time - signal_start_time).total_seconds() >= MAX_SIGNAL_LENTH:
                                print("Max signal length reached, stopping detection.")
                                finish_detection(p, signals_count, stream)
                                return
                            signal_finish_time = now
                            break
                        print(f"Signal {target_freq} Hz detected!")
                        if signals_count < 2:
                            signals_count += 1
                        print(signals_count)
                        signal_start_time = signal_finish_time = now
                        break
            else:
                if (signal_finish_time != signal_start_time) and (signal_finish_time - signal_start_time).total_seconds() < MIN_SIGNAL_DURATION:
                    print("Signal lost, resetting count.")
                    if signals_count > 0:
                        signals_count -= 1
                    else:
                        signals_count = 0
                    if signal_finish_time != signal_start_time:
                        signal_finish_time = signal_start_time
            if (now - signal_finish_time).total_seconds() > TIME_BUFFER and signals_count > 0:
                finish_detection(p, signals_count, stream)

                return

            mags_norm = mags / (np.max(mags) + 1e-9)
            line.set_data(xf, mags_norm)
            ax.set_xlim(0, min(5000, RATE/2)) 
            ax.set_ylim(0, 1.05)
            text.set_text(f"Dominant: {dom_freq:.2f} Hz" if dom_freq else "Dominant: -")
            fig.canvas.draw()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("Interrupted by user")

    print("Stopping stream...(end of main)")
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Done.")

if __name__ == "__main__":
    main()
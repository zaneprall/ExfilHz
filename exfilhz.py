import numpy as np
import soundfile as sf
import argparse
from scipy.signal import butter, lfilter
from scipy.fftpack import fft
from pydub import AudioSegment
import logging
import sys
import os
from multiprocessing import Pool

def modulate_hex(data, tone_freqs, sample_rate, duration_per_symbol):
    # Convert binary data to hexadecimal
    hex_data = ''.join(format(byte, '02x') for byte in data)

    # Generate time vector
    t = np.arange(0, duration_per_symbol, 1/sample_rate)
    signal = np.array([])

    for hex_digit in hex_data:
        # Get the frequency for the current hex digit
        freq = tone_freqs[int(hex_digit, 16)]

        # Generate the tone
        tone = np.sin(2 * np.pi * freq * t)

        # Append the tone to the signal
        signal = np.concatenate((signal, tone))

    signal = signal.astype(np.float32)  # Ensure the data type is float32
    max_val = np.max(np.abs(signal))    # Find the maximum absolute value for normalization
    if max_val > 0:
        signal = signal / max_val       # Normalize

    return signal

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def demodulate_hex(signal, tone_freqs, sample_rate, duration_per_symbol):
    # Apply band-pass filter
    lowcut = min(tone_freqs) - 100  # Adjust buffer range as needed
    highcut = max(tone_freqs) + 100
    signal = butter_bandpass_filter(signal, lowcut, highcut, sample_rate, order=6)

    t = np.arange(0, duration_per_symbol, 1/sample_rate)
    decoded_hex = ''

    for i in range(0, len(signal), len(t)):
        symbol_segment = signal[i:i+len(t)]
        # Compute FFT of the symbol segment
        symbol_freqs = fft(symbol_segment)
        # Find the peak frequency
        peak_freq_idx = np.argmax(np.abs(symbol_freqs[:len(symbol_freqs)//2]))
        peak_freq = peak_freq_idx * sample_rate / len(t)

        # Find the closest tone frequency
        closest_tone_idx = np.argmin(np.abs(np.array(tone_freqs) - peak_freq))
        decoded_hex += format(closest_tone_idx, 'x')

    decoded_bytes = bytes.fromhex(decoded_hex)
    return decoded_bytes

def process_file(input_file, output_file, tone_freqs, sample_rate, duration_per_symbol):
    # Determine the operation based on file extension
    is_decode = input_file.lower().endswith(('.flac', '.wav', '.mp3'))

    if is_decode:
        # Decoding: Read the audio file and demodulate the signal
        if input_file.lower().endswith(('.flac', '.wav')):
            signal, _ = sf.read(input_file)
        elif input_file.lower().endswith('.mp3'):
            audio_segment = AudioSegment.from_mp3(input_file)
            signal = np.array(audio_segment.get_array_of_samples())
        else:
            logging.error(f'Unsupported input file format for decoding: {input_file}')
            sys.exit(1)

        # Demodulate the signal to retrieve data
        data = demodulate_hex(signal, tone_freqs, sample_rate, duration_per_symbol)

        # Write the retrieved data to the output file
        with open(output_file, 'wb') as f:
            f.write(data)
        logging.info(f'Decoding complete: {input_file} -> {output_file}')

    else:
        # Encoding: Read the input file and modulate the data
        with open(input_file, 'rb') as f:
            data = f.read()

        signal = modulate_hex(data, tone_freqs, sample_rate, duration_per_symbol)

        # Write the modulated signal to an audio file
        if output_file.lower().endswith('.flac') or output_file.lower().endswith('.wav'):
            sf.write(output_file, signal, sample_rate)
        elif output_file.lower().endswith('.mp3'):
            audio_segment = AudioSegment(
                signal.tobytes(),
                frame_rate=sample_rate,
                sample_width=signal.dtype.itemsize,
                channels=1
            )
            audio_segment.export(output_file, format="mp3")
        else:
            logging.error('Unsupported output file format for encoding.')
            sys.exit(1)

        logging.info(f'Encoding complete: {input_file} -> {output_file}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert between files and audio using hexadecimal encoding. '
                    'The script uses a single modulation scheme based on hexadecimal tones for encoding and decoding data into audio signals. '
                    'This supports infrasonic and ultrasonic audio to be ouside the human range of hearing for stealthy exfiltration. '
                    'Lowest and highest values can be assigned to best fit the hardware you have access to.'
                    'Some soundwaves penetrate surfaces such as glass better than others. try a lower range of freqencies for material penetration. '
                    'Please keep in mind the default options are ultrasonic, so do not expect it to be audible without adjusting the -L and -H variables. '
                    'Files may take a long time to generate. ',

        usage='%(prog)s INPUT OUTPUT.flac [options]'
    )

    parser.add_argument('input', help='Path to the input file or the input FLAC audio.')
    parser.add_argument('output', help='Path for the output file or the output FLAC audio.')
    parser.add_argument('-r', '--rate', type=int, default=48000, help='Sample rate in Hz (default: 48000).')
    parser.add_argument('-L', '--lowest', type=int, default=22000, help='Lowest frequency of the range in Hz (default: 22000).')
    parser.add_argument('-H', '--highest', type=int, default=24000, help='Highest frequency of the range in Hz (default: 24000).')
    parser.add_argument('-d', '--duration', type=float, default=0.1, help='Duration per symbol in seconds. This will impact the length of the .flac file. (default: 0.1).')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output for debugging and detailed process information.')
    parser.add_argument('-p', '--processes', type=int, default=os.cpu_count(), help='Number of processes to use (default: use all available cores).')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Calculate step size and generate tone frequencies
    step_size = (args.highest - args.lowest) / 15
    tone_freqs = [args.lowest + i * step_size for i in range(16)]

    # Validate the frequency range and rate
    if args.highest >= args.rate / 2:
        logging.error(f"The highest frequency ({args.highest} Hz) must be less than half the sample rate (Nyquist limit). Current sample rate: {args.rate} Hz")
        sys.exit(1)

    if args.lowest >= args.highest or args.lowest < 0:
        logging.error(f"The lowest frequency must be non-negative and less than the highest frequency. Lowest: {args.lowest} Hz, Highest: {args.highest} Hz")
        sys.exit(1)

    # List of files to process (modify according to your use case)
    files_to_process = [(args.input, args.output)]

    # Parameters constant for all tasks
    constant_params = (tone_freqs, args.rate, args.duration)

    # Use multiprocessing.Pool to parallelize the file processing
    with Pool(processes=args.processes) as pool:
    # Prepare the arguments for each call to process_file
        tasks = [(args.input, args.output) + constant_params for _ in range(args.processes)]

        # Map process_file function to each set of arguments
        pool.starmap(process_file, tasks)

if __name__ == "__main__":
    main()

# ExfilHz: Covert Data Transfer via Audiofiles
### Description

ExfilHz is a tool designed for covert data transfer using audio signals. The tool leverages hexadecimal encoding to convert files into audio and vice versa. It supports infrasonic and ultrasonic frequencies, enabling operations outside the human range of hearing for stealthy exfiltration.

You can customize the frequency range to suit your hardware capabilities and the specific requirements of your environment. The tool is especially useful for scenarios where conventional data transfer methods are not feasible or are heavily monitored.
### Features:

- Hexadecimal Encoding: Converts files to and from audio using a hexadecimal-based modulation scheme.
- Infrasonic and Ultrasonic Support: Operates outside the human range of hearing to avoid detection.
- Customizable Frequency Range: Tailor the frequency range to match your hardware and penetration needs.
- Material Penetration: Lower frequency ranges can be used to penetrate materials like glass more effectively.
- supports .flac, .wav, and .mp3
### Limitations:

- Processing Time: Generating files, especially large ones, may take a significant amount of time.
- Hardware Dependency: The effectiveness of frequency penetration and the quality of audio conversion depend on the hardware used for playback and recording.
- Audibility: Default settings use ultrasonic frequencies. Adjust -L and -H for audible ranges if needed.
- Interference: Noisy environments will be considerably harder to exfiltrate data from. You may need to transmit multiple times and normalize the data. 
### Requirements

Ensure you have the following requirements installed:

    Python 3.x
    NumPy
    SciPy
    SoundFile
    pydub

Pip command:

    pip install -r requirements.txt

You'll likely need to add ffmpeg to your environment variables if building a file on windows. This tool is significantly easier to get running on linux.

### Usage
Arguments

    input: Path to the input file or the input FLAC audio.
    output: Path for the output file or the output FLAC audio.
    -r / --rate: Sample rate in Hz (default: 48000).
    -L / --lowest: Lowest frequency of the range in Hz (default: 22000).
    -H / --highest: Highest frequency of the range in Hz (default: 24000).
    -d / --duration: Duration per symbol in seconds (default: 0.1).
    -v / --verbose: Enable verbose output for debugging and detailed process information.

### Running the Tool

Convert a file to an audio signal:

    python exfilhz.py input_file output_audio.flac --lowest 18000 --highest 22000

Convert an audio signal back to a file:

    python exfilhz.py input_audio.flac output_file --lowest 18000 --highest 22000

Contributing

Contributions to ExfilHz are welcome. Feel free to fork the repository, make your changes, and submit a pull request. changes to make this more efficient would be most beneficial. I need to implement chunking to evenly divide the input file for each subprocess.

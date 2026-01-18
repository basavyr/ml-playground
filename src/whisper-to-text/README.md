# Whisper to Text

A command-line tool for real-time speech-to-text transcription using OpenAI's Whisper models.

## How It Works

This script records audio from your microphone, saves it as a temporary WAV file, and then uses a speech-to-text engine to transcribe the audio. The resulting text is displayed in the console and automatically copied to your clipboard.

You can choose between two transcription engines:
- **`openai-whisper`**: The default, high-quality engine (requires a GPU for optimal performance).
- **`whisper.cpp`**: A CPU-optimized version that is generally faster on machines without a powerful GPU.

## How to Use

### 1. Installation

First, install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Configuration

Before the first run, configure the tool to use your preferred audio device and model size:
```bash
python configure.py
```
You will be prompted to select a microphone and a Whisper model.

### 3. Running the Transcriber

To start the application, run:
```bash
python whisper_transcriber.py
```
- Press `Enter` to start recording.
- Press `Enter` again to stop recording and begin transcription.

To use the CPU-optimized `whisper.cpp` engine, use the `--pycpp` flag:
```bash
python whisper_transcriber.py --pycpp
```

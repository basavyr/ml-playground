import sys
import pyaudio
import wave
import os
import json
import threading
import time

try:
    import pyperclip
except ImportError:
    pyperclip = None
    print("Warning: 'pyperclip' not found. Install it for clipboard functionality: pip install pyperclip")

# --- Configuration ---
WAVE_OUTPUT_FILENAME = "temp_recording.wav"
CONFIG_FILE = "config.json"
CHANNELS = 1
# RATE will be set dynamically based on the chosen engine
RATE = 44100  # Default
CHUNK = 1024
FORMAT = pyaudio.paInt16

def load_config():
    """Loads device configuration from the JSON file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

class AudioRecorder:
    def __init__(self, audio_instance, rate, device_index=None):
        self.audio = audio_instance
        self.rate = rate
        self.device_index = device_index
        self.frames = []
        self.is_recording = False
        self.stream = None
        self.thread = None

    def start_recording(self):
        self.frames = []
        self.is_recording = True
        try:
            self.stream = self.audio.open(format=FORMAT,
                                           channels=CHANNELS,
                                           rate=self.rate,
                                           input=True,
                                           input_device_index=self.device_index,
                                           frames_per_buffer=CHUNK)
        except IOError:
            print(f"\nWarning: Audio device at index {self.device_index} is not available. Falling back to default.")
            print("Run 'python configure.py' to select a valid device.\n")
            self.device_index = None  # Fallback to default
            self.stream = self.audio.open(format=FORMAT,
                                           channels=CHANNELS,
                                           rate=self.rate,
                                           input=True,
                                           input_device_index=self.device_index,
                                           frames_per_buffer=CHUNK)
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.start()

    def _record_loop(self):
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
            except IOError:
                pass

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.thread:
            self.thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.stream = None
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

def transcribe_openai(model, filename):
    """Transcribes audio using the openai-whisper library."""
    result = model.transcribe(filename)
    return result["text"]

def transcribe_cpp(model, filename):
    """Transcribes audio using the pywhispercpp library."""
    result = model.transcribe(filename, n_threads=4)
    # pywhispercpp returns a list of segments
    return "".join([segment.text for segment in result])

def main():
    use_cpp = "--pycpp" in sys.argv

    config = load_config()
    model_type = config.get("model_type", "base.en")
    device_index = config.get("device_index")
    device_name = config.get("device_name")

    # --- Dynamically set engine and parameters ---
    model = None
    transcribe_func = None
    global RATE

    if use_cpp:
        engine_name = "whisper.cpp (CPU)"
        RATE = 16000  # whisper.cpp requires 16kHz
        try:
            print(f"Loading '{model_type}' model with {engine_name}. (Pass '--pycpp' to switch)")
            model = Model(model_type, n_threads=4)
            transcribe_func = transcribe_cpp
        except ImportError:
            print("ERROR: 'pywhispercpp' is not installed.")
            print("Please install it with: pip install pywhispercpp")
            return
        except Exception as e:
            print(f"Error loading whisper.cpp model: {e}")
            return
    else:
        engine_name = "openai-whisper (CPU/GPU)"
        RATE = 44100  # Original library is more flexible
        try:
            import whisper
            import warnings
            warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
            print(f"Loading '{model_type}' model with {engine_name}. (Pass '--pycpp' to switch)")
            model = whisper.load_model(model_type)
            transcribe_func = transcribe_openai
        except ImportError:
            print("ERROR: 'openai-whisper' is not installed.")
            print("Please install it with: pip install openai-whisper")
            return
        except Exception as e:
            print(f"Error loading openai-whisper model: {e}")
            return
            
    p = pyaudio.PyAudio()
    recorder = AudioRecorder(p, rate=RATE, device_index=device_index)

    if device_name:
        print(f"(Using audio device: '{device_name}'. Run 'python configure.py' to change.)")

    try:
        while True:
            input("\nPress Enter to record (then Enter again to stop)... ")
            recorder.start_recording()
            print("Recording...")
            
            input() # Wait for user to stop
            recorder.stop_recording()
            print("Recording stopped. Transcribing...")
            try:
                start_time = time.time()
                transcribed_text = transcribe_func(model, WAVE_OUTPUT_FILENAME)
                end_time = time.time()
                transcription_time = end_time - start_time
                print(transcribed_text.strip())
                print() # Blank line
                info_message = f"(Transcription took {transcription_time:.2f} seconds"
                if pyperclip:
                    pyperclip.copy(transcribed_text.strip())
                    info_message += ". Copied to clipboard"
                info_message += ")"
                print(info_message)
            except Exception as e:
                print(f"An error occurred during transcription: {e}")
            finally:
                if os.path.exists(WAVE_OUTPUT_FILENAME):
                    os.remove(WAVE_OUTPUT_FILENAME)

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if recorder.is_recording:
            recorder.stop_recording()
        p.terminate()
        print("\nExiting.")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
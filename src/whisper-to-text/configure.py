import pyaudio
import json
import os

CONFIG_FILE = "config.json"

def choose_device(p, current_config):
    """
    Lists available audio input devices and returns the user's choice.
    """
    print("Searching for available audio input devices...")

    input_devices = []
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:
            input_devices.append((i, device_info.get('name')))

    if not input_devices:
        print("Error: No audio input devices found.")
        return None, None

    print("\n--- Step 1: Select Audio Device ---")
    for i, (device_index, device_name) in enumerate(input_devices):
        print(f"  [{i}] {device_name}")

    current_device_name = current_config.get('device_name', 'None')
    print(f"\nCurrent default device: {current_device_name}\n")

    while True:
        try:
            choice_str = input("Enter the number of the device you want to use: ")
            if not choice_str: # Allow user to skip by pressing Enter
                return current_config.get('device_index'), current_config.get('device_name')
            choice = int(choice_str)
            if 0 <= choice < len(input_devices):
                return input_devices[choice]
            else:
                print(f"Invalid selection. Please enter a number between 0 and {len(input_devices) - 1}.")
        except (ValueError, TypeError):
            print("Invalid input. Please enter a number, or press Enter to keep the current setting.")
        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled.")
            return None, None

def choose_model(current_config):
    """
    Lists available Whisper models and returns the user's choice.
    """
    models = {
        'tiny.en': 'Fastest, lowest accuracy',
        'base.en': 'Faster, low accuracy',
        'small.en': 'Balanced speed and accuracy',
        'medium.en': 'Slower, high accuracy',
        'large': 'Slowest, highest accuracy (multilingual, but best for English)',
    }
    
    print("\n--- Step 2: Select Whisper Model ---")
    model_keys = list(models.keys())
    for i, model_name in enumerate(model_keys):
        print(f"  [{i}] {model_name:<10} - {models[model_name]}")

    current_model = current_config.get('model_type', 'base.en')
    print(f"\nCurrent default model: {current_model}\n")

    while True:
        try:
            choice_str = input("Enter the number of the model you want to use: ")
            if not choice_str: # Allow user to skip by pressing Enter
                return current_model
            choice = int(choice_str)
            if 0 <= choice < len(model_keys):
                return model_keys[choice]
            else:
                print(f"Invalid selection. Please enter a number between 0 and {len(model_keys) - 1}.")
        except (ValueError, TypeError):
            print("Invalid input. Please enter a number, or press Enter to keep the current setting.")
        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled.")
            return None


def main():
    p = pyaudio.PyAudio()

    # Load existing config or create an empty one
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                pass # Use empty config if file is malformed

    # Step 1: Choose Device
    selected_device_index, selected_device_name = choose_device(p, config)
    if selected_device_name is None:
        p.terminate()
        print("Exiting configuration.")
        return

    # Step 2: Choose Model
    selected_model = choose_model(config)
    if selected_model is None:
        p.terminate()
        print("Exiting configuration.")
        return

    # Update the configuration dictionary
    config['device_index'] = selected_device_index
    config['device_name'] = selected_device_name
    config['model_type'] = selected_model
    
    # Save the updated configuration
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

    print("\n--- Configuration Saved ---")
    print(f"Device: '{config['device_name']}'")
    print(f"Model:  '{config['model_type']}'")
    print(f"Settings saved to '{CONFIG_FILE}'.")
    
    p.terminate()


if __name__ == "__main__":
    main()

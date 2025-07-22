import sounddevice as sd
import soundfile as sf
import numpy as np
import requests
import os

# Audio recording parameters
SAMPLE_RATE = 44100  # Hz (match your model's expected sample rate)
CHANNELS = 1  # Mono
DURATION = 5  # Seconds
OUTPUT_FILE = "audio.wav"
API_URL = "http://127.0.0.1:5000/test"

def record_audio():
    print("Recording...")
    # Record audio
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    
    # Save to WAV file
    sf.write(OUTPUT_FILE, audio, SAMPLE_RATE, subtype='PCM_16')  # PCM_16 for compatibility
    return OUTPUT_FILE

def send_audio_to_api(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        files = {'audio': (audio_file_path, audio_file, 'audio/wav')}
        try:
            response = requests.post(API_URL, files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to send audio to API: {str(e)}"}

def main():
    try:
        # Record and save audio
        audio_file = record_audio()
        
        # Send to API
        api_response = send_audio_to_api('angry.wav')
        print("API Response:", api_response)
        
    finally:
        # Clean up temporary file
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)

if __name__ == "__main__":
    main()
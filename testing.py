import os
import requests
import pyttsx3
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # or hardcode your key here

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Load Whisper model (base is fast)
model = WhisperModel("base")

# Configure pyttsx3 TTS engine for better voice output
tts_engine.setProperty('rate', 150)  # Adjust speech rate (words per minute)
tts_engine.setProperty('volume', 1)  # Set volume level (0.0 to 1.0)

# Record microphone input
def record_audio(filename="input.wav", duration=5, fs=16000):
    print("🎤 Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, audio)
    print("✅ Recording complete.")


def transcribe_audio(filename="input.wav"):
    segments, _ = model.transcribe(filename)
    transcript = " ".join([seg.text for seg in segments])
    print(f"📝 You said: {transcript}")
    return transcript.strip()


def ask_openrouter(prompt, model="openai/gpt-3.5-turbo"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        print(f"🤖 Assistant: {reply}")
        return reply
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return "Sorry, I couldn't reach the assistant."


def speak(text):
    tts_engine.say(text)  # Speak the text
    tts_engine.runAndWait()  # Wait until the speaking is finished


if __name__ == "__main__":
    print("Say 'stop' anytime to end the conversation.")
    while True:
        record_audio(duration=5)  # Start by listening
        question = transcribe_audio()  # Convert the audio to text

        if not question:
            print("⚠️ No valid speech detected, please try again.")
            continue  # Skip if no valid question is detected

        if "stop" in question.lower():
            print("🛑 Stopping the conversation. Goodbye!")
            speak("Goodbye!")  # Speak "Goodbye" before stopping
            break  # Exit the loop to end the conversation

        answer = ask_openrouter(question)  # Get the answer from OpenRouter
        speak(answer)  # Speak the answer

        time.sleep(1)  # Small delay to ensure speech is finished before listening again

import os
import wave
import json
from vosk import Model, KaldiRecognizer
from moviepy.editor import VideoFileClip

VOSK_MODEL = os.path.join(os.path.dirname(__file__), "vosk_models/large_pt_br_model")

def load_dictionary(path="words.txt"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]
    return []

def mp4_to_wav(mp4_path, wav_path="audio.wav"):
    try:
        video = VideoFileClip(mp4_path)
        audio = video.audio
        audio.write_audiofile(
            wav_path,
            codec='pcm_s16le',
            ffmpeg_params=["-ac", "1", "-ar", "16000"]
        )
        video.close()
        return True
    except Exception as e:
        print(f"Conversion error: {str(e)}")
        return False

def transcribe_with_vosk(wav_path, model_path):
    if not os.path.exists(model_path):
        print(f"Vosk model not found at: {model_path}")
        return None

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    custom_words = load_dictionary()
    if custom_words:
        recognizer.SetWords(True)
        recognizer.SetPartialWords(True)

    try:
        with wave.open(wav_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                print("Incompatible audio format! Must be: Mono, 16kHz, 16-bit")
                return None

            full_text = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    full_text.append(result.get("text", ""))

            final_result = json.loads(recognizer.FinalResult())
            full_text.append(final_result.get("text", ""))

            return " ".join(full_text).strip()

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return None

if __name__ == "__main__":
    video_file = "your_video.mp4"

    if mp4_to_wav(video_file):
        print("Transcribing with Vosk...")
        transcript = transcribe_with_vosk("audio.wav", VOSK_MODEL)

        if transcript:
            with open("transcription.txt", "w") as f:
                f.write(transcript)
            print("Transcription completed successfully!")
        else:
            print("Transcription failed. Check logs.")
    else:
        print("Error in MP4 → WAV conversion")

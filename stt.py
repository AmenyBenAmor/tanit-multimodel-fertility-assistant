# ============================
# IMPORTS
# ============================
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Whisper non installé. Installez avec: pip install openai-whisper")

# Transformers (optionnel)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ============================
# CLASSE SpeechToText
# ============================
class SpeechToText:
    """Speech-to-Text avec reconnaissance des termes médicaux"""
    def __init__(
        self,
        model_type: str = "whisper",
        model_size: str = "base",
        language: str = "en",
        device: str = "cpu"
    ):
        self.model_type = model_type
        self.model_size = model_size
        self.language = language
        self.device = device
        self.model = None
        
        self.medical_vocabulary = [
            "AMH", "FSH", "LH", "estradiol", "progesterone", "testosterone",
            "PCOS", "endometriosis", "infertility", "IVF", "IUI",
            "ovulation", "follicle", "ultrasound", "ng/mL", "mIU/mL",
            "menstrual", "cycle", "ovarian", "hormone", "fertility"
        ]
        
        print(f"Initializing STT Module | Backend: {model_type}, Model: {model_size}, Language: {language}, Device: {device}")
        self._load_model()

    def _load_model(self):
        if self.model_type == "whisper":
            if not WHISPER_AVAILABLE:
                raise RuntimeError("Whisper non disponible. pip install openai-whisper")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print("Whisper model loaded successfully")
        elif self.model_type == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers non disponible. pip install transformers")
            model_name = "openai/whisper-tiny" if self.model_size=="tiny" else "openai/whisper-base"
            self.model = pipeline("automatic-speech-recognition", model=model_name, device=0 if self.device=="cuda" else -1)
            print("Transformers ASR loaded successfully")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def transcribe_file(self, audio_path: str, initial_prompt: Optional[str]=None) -> Dict[str, Any]:
        if not os.path.exists(audio_path):
            return {"success": False, "error": f"File not found: {audio_path}"}
        if self.model_type=="whisper":
            return self._transcribe_whisper(audio_path, initial_prompt)
        else:
            return self._transcribe_transformers(audio_path)

    def _transcribe_whisper(self, audio_path: str, initial_prompt: Optional[str]=None):
        if initial_prompt is None:
            initial_prompt = "Medical conversation about fertility, hormones, AMH, FSH, PCOS, ovulation, and reproductive health."
        result = self.model.transcribe(audio_path, language=self.language, initial_prompt=initial_prompt, verbose=False)
        transcription = self._post_process_medical_terms(result["text"].strip())
        return {"success": True, "text": transcription, "language": result.get("language", self.language)}

    def _transcribe_transformers(self, audio_path: str):
        result = self.model(audio_path)
        transcription = self._post_process_medical_terms(result["text"].strip())
        return {"success": True, "text": transcription, "language": self.language}

    def _post_process_medical_terms(self, text: str) -> str:
        replacements = {
            "a m h": "AMH", "a. m. h.": "AMH",
            "f s h": "FSH", "f. s. h.": "FSH",
            "l h": "LH",
            "p c o s": "PCOS", "p. c. o. s.": "PCOS",
            "i v f": "IVF", "i. v. f.": "IVF",
            "i u i": "IUI",
            "nanogram per milliliter": "ng/mL",
            "nanograms per milliliter": "ng/mL",
            "milli international unit": "mIU/mL",
        }
        text_lower = text.lower()
        for wrong, correct in replacements.items():
            text_lower = text_lower.replace(wrong, correct)
        return text_lower

# ============================
# FONCTION DÉMO
# ============================
def demo_stt(audio_file="test_audio.wav"):
    stt = SpeechToText(model_type="whisper", model_size="base", language="en", device="cpu")
    if os.path.exists(audio_file):
        result = stt.transcribe_file(audio_file, initial_prompt="Medical discussion about AMH levels and PCOS")
        if result["success"]:
            print("\nTranscription réussie :")
            print(f"   Text: {result['text']}")
        else:
            print(f"\nErreur: {result['error']}")
    else:
        print(f"\nAudio file not found: {audio_file}")

# ============================
# ENREGISTREMENT MICROPHONE DYNAMIQUE
# ============================
def record_microphone_manual(filename="test_audio.wav", fs=16000):
    print("🎤 Appuie sur Entrée pour commencer l'enregistrement...")
    input()
    print("🎙️ Enregistrement en cours... Appuie sur Entrée pour stopper.")

    audio_data = []

    def callback(indata, frames, time, status):
        audio_data.append(indata.copy())

    stream = sd.InputStream(samplerate=fs, channels=1, callback=callback)
    stream.start()

    input()  # Arrêt de l'enregistrement
    stream.stop()
    stream.close()

    audio_array = np.concatenate(audio_data, axis=0)
    write(filename, fs, audio_array)
    print(f"Audio enregistré : {filename}")

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    # 1️⃣ Enregistrement micro dynamique
    record_microphone_manual(filename="test_audio.wav")

    # 2️⃣ Transcription
    demo_stt(audio_file="test_audio.wav")

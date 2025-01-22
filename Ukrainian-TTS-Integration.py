import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
from bark import SAMPLE_RATE, generate_audio
from ukrainian_tts.tts import TTS

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Synthesizer(Enum):
    UKRAINIAN_TTS = "Ukrainian-TTS"
    BARK = "Bark"
    RHVOICE = "RHVoice"

class StressType(Enum):
    DICTIONARY = "dictionary"
    MODEL = "model"

class Voice(Enum):
    TETIANA = "tetiana"
    MYKYTA = "mykyta"
    LADA = "lada"
    DMYTRO = "dmytro"
    OLEKSA = "oleksa"

@dataclass
class TTSConfig:
    """Конфігурація для синтезу мовлення"""
    output_dir: Path = Path("output")
    sample_rate: int = SAMPLE_RATE
    device: str = "cpu"

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

class TTSEngine:
    """Головний клас для синтезу мовлення"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self._ukrainian_tts: Optional[TTS] = None
        self.rhvoice_voice_map = {
            Voice.TETIANA: "Anna",
            Voice.MYKYTA: "Ivan",
            Voice.LADA: "Maria",
            Voice.DMYTRO: "Pavel",
            Voice.OLEKSA: "Aleksandr"
        }

    @property
    def ukrainian_tts(self) -> TTS:
        """Ліниве завантаження Ukrainian-TTS"""
        if self._ukrainian_tts is None:
            self._ukrainian_tts = TTS(device=self.config.device)
        return self._ukrainian_tts

    def synthesize(
        self,
        text: str,
        synthesizer: Synthesizer,
        voice: Voice,
        stress: StressType
    ) -> Tuple[Optional[str], str]:
        """
        Синтезує мовлення використовуючи вказаний синтезатор.
        
        Returns:
            Tuple[Optional[str], str]: (шлях до аудіофайлу, текст результату)
        """
        try:
            method = getattr(self, f"_synthesize_{synthesizer.value.lower()}")
            return method(text, voice, stress)
        except AttributeError:
            return None, f"Непідтримуваний синтезатор: {synthesizer}"
        except Exception as e:
            logger.exception(f"Помилка синтезу {synthesizer}")
            return None, f"Помилка синтезу: {str(e)}"

    def _synthesize_ukrainian_tts(
        self,
        text: str,
        voice: Voice,
        stress: StressType
    ) -> Tuple[str, str]:
        output_file = self.config.output_dir / "ukrainian_tts_output.wav"
        
        with open(output_file, mode="wb") as file:
            _, accented_text = self.ukrainian_tts.tts(
                text, voice.value, stress.value, file
            )
        
        return str(output_file), f"Текст з наголосами: {accented_text}"

    def _synthesize_bark(
        self,
        text: str,
        voice: Voice,
        stress: StressType
    ) -> Tuple[str, str]:
        output_file = self.config.output_dir / "bark_output.wav"
        
        audio_array = generate_audio(text)
        sf.write(output_file, audio_array, self.config.sample_rate)
        
        return str(output_file), "Синтезовано за допомогою Bark!"

    def _synthesize_rhvoice(
        self,
        text: str,
        voice: Voice,
        stress: StressType
    ) -> Tuple[str, str]:
        import subprocess
        
        output_file = self.config.output_dir / "rhvoice_output.wav"
        rhvoice_voice = self.rhvoice_voice_map.get(voice, "Anna")
        
        command = [
            "RHVoice-client",
            "-p", rhvoice_voice,
            "-o", str(output_file)
        ]
        
        process = subprocess.run(
            command,
            input=text.encode("utf-8"),
            capture_output=True
        )
        
        if process.returncode == 0:
            return str(output_file), "Синтезовано за допомогою RHVoice!"
        else:
            error = process.stderr.decode("utf-8")
            raise RuntimeError(f"Помилка RHVoice: {error}")

class GradioInterface:
    """Клас для роботи з Gradio інтерфейсом"""
    
    def __init__(self, tts_engine: TTSEngine):
        self.tts_engine = tts_engine
        self.interface = self._create_interface()

    def _create_interface(self) -> gr.Interface:
        return gr.Interface(
            fn=self._process_tts_request,
            inputs=[
                gr.Textbox(
                    label="Текст для озвучки",
                    placeholder="Введіть текст тут"
                ),
                gr.Dropdown(
                    choices=[v.value for v in Voice],
                    label="Голос",
                    value=Voice.DMYTRO.value
                ),
                gr.Dropdown(
                    choices=[s.value for s in StressType],
                    label="Тип наголосу (тільки для Ukrainian-TTS)",
                    value=StressType.DICTIONARY.value
                ),
                gr.Radio(
                    choices=["cpu", "gpu", "mps"],
                    label="Пристрій (тільки для Ukrainian-TTS)",
                    value="cpu"
                ),
                gr.Dropdown(
                    choices=[s.value for s in Synthesizer],
                    label="Синтезатор",
                    value=Synthesizer.UKRAINIAN_TTS.value
                )
            ],
            outputs=[
                gr.Textbox(label="Результат синтезу"),
                gr.Audio(label="Результуючий аудіофайл")
            ],
            title="Український TTS + Bark + RHVoice",
            description="Синтез мовлення українською мовою з вибором між Ukrainian-TTS, Bark та RHVoice."
        )

    def _process_tts_request(
        self,
        text: str,
        voice: str,
        stress: str,
        device: str,
        synthesizer: str
    ) -> Tuple[str, Optional[str]]:
        """Обробка запиту на синтез мовлення"""
        try:
            voice_enum = Voice(voice.lower())
            stress_enum = StressType(stress.lower())
            synthesizer_enum = Synthesizer(synthesizer)
            
            self.tts_engine.config.device = device
            
            return self.tts_engine.synthesize(
                text,
                synthesizer_enum,
                voice_enum,
                stress_enum
            )
        except ValueError as e:
            return str(e), None

    def launch(self, **kwargs):
        """Запуск Gradio інтерфейсу"""
        self.interface.launch(**kwargs)

def main():
    """Головна функція програми"""
    config = TTSConfig()
    tts_engine = TTSEngine(config)
    interface = GradioInterface(tts_engine)
    
    logger.info("Запуск веб-інтерфейсу...")
    interface.launch(share=True)

if __name__ == "__main__":
    main()

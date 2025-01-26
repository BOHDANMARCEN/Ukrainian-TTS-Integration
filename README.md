Веб-інтерфейс для синтезу українського мовлення з підтримкою декількох движків: Ukrainian-TTS, Bark та RHVoice.

## 🌟 Особливості

- 🔊 Підтримка трьох систем синтезу мовлення:
  - Ukrainian-TTS
  - Bark
  - RHVoice
- 👥 П'ять різних голосів: Тетяна, Микита, Лада, Дмитро, Олекса
- 🎯 Два типи наголосів для Ukrainian-TTS: словниковий та модельний
- 💻 Підтримка CPU/GPU/MPS для прискорення синтезу
- 🌐 Зручний веб-інтерфейс на базі Gradio

## 📋 Вимоги

- Python 3.8+
- Git
- RHVoice (опціонально)

## 🚀 Встановлення

1. Клонуйте репозиторій:
```bash
git clone https://github.com/your-username/ukrainian-multi-tts.git
cd ukrainian-multi-tts
```

2. Встановіть необхідні пакети:
```bash
pip install git+https://github.com/robinhad/ukrainian-tts.git gradio bark
```

3. (Опціонально) Встановіть RHVoice:
```bash
# Ubuntu/Debian
sudo apt-get install rhvoice

# MacOS
brew install rhvoice

# Windows
# Завантажте інсталятор з офіційного сайту RHVoice
```

## 💻 Використання

1. Запустіть програму:
```bash
python main.py
```

2. Відкрийте веб-браузер за посиланням, яке з'явиться в консолі (зазвичай http://localhost:7860)

3. У веб-інтерфейсі:
   - Введіть текст для синтезу
   - Виберіть голос
   - Виберіть тип наголосу (для Ukrainian-TTS)
   - Виберіть пристрій для обчислень
   - Виберіть синтезатор мовлення
   - Натисніть "Submit"

## 🔧 Конфігурація

Ви можете налаштувати параметри синтезу в класі `TTSConfig`:

```python
@dataclass
class TTSConfig:
    output_dir: Path = Path("output")  # Директорія для збереження аудіофайлів
    sample_rate: int = SAMPLE_RATE     # Частота дискретизації
    device: str = "cpu"                # Пристрій для обчислень (cpu/gpu/mps)
```

## 📦 Структура проекту

```
ukrainian-multi-tts/
├── main.py           # Головний файл програми
├── output/           # Директорія для згенерованих аудіофайлів
├── README.md         # Документація
└── requirements.txt  # Залежності проекту
```

## 🤝 Внесок

Будь ласка, не соромтесь створювати issues та pull requests. Для значних змін, спочатку створіть issue для обговорення пропонованих змін.

## 📄 Ліцензія

Цей проект ліцензований під MIT License - дивіться файл [LICENSE](LICENSE) для деталей.

## ⭐ Подяки

- [Ukrainian-TTS](https://github.com/robinhad/ukrainian-tts) за базову бібліотеку синтезу українського мовлення
- [Bark](https://github.com/suno-ai/bark) за універсальний синтезатор мовлення
- [RHVoice](https://github.com/RHVoice/RHVoice) за відкритий синтезатор мовлення
- [Gradio](https://gradio.app/) за інструменти для створення веб-інтерфейсу

## 📞 Контакти

Якщо у вас є питання або пропозиції, будь ласка:
1. Створіть issue в цьому репозиторії
2. Відправте pull request
3. Зв'яжіться зі мною через:
     whatsapp +48731485019
     metfarell@gmail.com]

# Vocale
A speech library designed specifically for personal assistants. It features a wake word recognizer, voice activity detection, offline mode, and a speech synthesizer.

Vocale simply combines the following libraries and technologies into one:
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
- [Vosk](https://github.com/alphacep/vosk-api)
- [PVPorcupine](https://pypi.org/project/pvporcupine/)
- [Pyttsx3](https://pypi.org/project/pyttsx3/)
- Voice Activity Detection

## Installation
```
pip install vocale-python
```

After installing, you are required to download the voice activity detection model.
It can be found inside the models folder of this repository.

Next just download one of the available vosk api models from this website. https://alphacephei.com/vosk/models

## Example
```py
import asyncio
from vocale import SpeechRecognizer

async def callback(data: dict, *args, **kwargs) -> None:
    print(data)

async def main(loop):
    recognizer = SpeechRecognizer(
        ["friday", "odis", "jarvis"], # Wake words
        [1.0], # Sensitivities corresponding to the wakewords that are inside pvporcupine.KEYWORDS
        "models/vad.h5", # Path to the voice activity detection model
        0.9, # Threshold for the voice activity detection. Normally you would just leave this at default
        "models/vosk", # Path to the vosk model you've downloaded
        ".tmp/cache.wav", # Path to save the .wav file of the recognized audio (used by the google recognizer)
        callback, # The callback function defined above
        loop # Asyncio event loop
    )

    # Start and block the program
    await recognizer.start(blocking=True)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
```

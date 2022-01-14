install_requires = [
    "tensorflow",
    "numpy",
    "pvporcupine==1.9.0",
    "speechrecognition",
    "vosk",
    "wave",
    "sounddevice",
    "soundfile",
    "sox",
    "pyttsx3",
    "fluxhelper"
]

from distutils.core import setup
setup(
    name="vocale-python",
    packages=["vocale"],
    version="0.5",
    license="MIT",
    description="A speech library designed specifically for personal assistants. It features a wake word recognizer, voice activity detection, offline mode, and a speech synthesizer.",
    author="Philippe Mathew",
    author_email="philmattdev@gmail.com",
    url="https://github.com/bossauh/vocale",
    download_url="https://github.com/bossauh/vocale/releases/tag/v_05",
    keywords=["speech", "recognition"],
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
    ]
)
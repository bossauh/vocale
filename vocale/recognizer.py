import asyncio
import json
import os
import time
import warnings
import wave
from queue import Queue
from typing import Callable

import numpy as np
import pvporcupine
import sounddevice as sd
import vosk
from fluxhelper import osInterface
from speech_recognition import AudioFile, Recognizer, UnknownValueError
from tensorflow.keras.models import load_model
from vosk import SetLogLevel

RATE = 16000
DURATION = 0.5
CHANNELS = 1
CHUNK = 512
MAX_FREQ = 18


# Disable logging
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SetLogLevel(-1)


class VAD:

    """
    Main Voice activity detection class.
    This uses deep learning to predict whether a piece of audio is considered a speech or not a speech.

    Parameters
    ----------
    `modelPath` : str
        path to the model.h5 file.
    `sensitivity : float
        how sensitive the detection is.

    Methods
    -------
    `isSpeech(stream: bytes)` :
        returns True if the classified stream is a voice and False if not.

    """

    def __init__(self, modelPath: str, sensitivity: float = 0.90):
        self.model = load_model(modelPath)

        self.buffer = []
        self.sensitivity = sensitivity

    async def _formatPredictions(self, predictions) -> list:

        """
        Format the predictions into a more readable and easy to traverse format.
        """

        predictions = [[i, float(r)] for i, r in enumerate(predictions)]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions

    async def isSpeech(self, stream: bytes) -> bool:

        """
        Makes a prediction from the given stream bytes.

        Parameters
        ----------
        `stream` : bytes
            raw bytes stream (usually retrieved from pyaudio's .read function or sounddevice)

        Returns True if the classified stream is a voice and False if not.
        """

        # Convert the raw streams into a numpy array and get the decibels
        arr = np.frombuffer(stream, dtype=np.int16)
        db = 20 * np.log10(np.abs(np.fft.rfft(arr[:2048])))

        # Collect decibel values from relevent frequencies (MAX_FREQ)
        features = list(np.round(db[3:MAX_FREQ], 2))
        self.buffer.append(features)

        if len(self.buffer) == int(RATE / CHUNK * DURATION):
            total = np.array([x for y in self.buffer for x in y])
            self.buffer.clear()

            # Make the prediction
            predictions = self.model(np.array([total]))[0]
            predictions = await self._formatPredictions(predictions)

            index, probability = predictions[0]
            if index == 1 and probability >= self.sensitivity:
                # 1 is the index of speech and 0 is non speech
                return True
        return False


class SpeechRecognizer:
    def __init__(
        self,
        wakewords: list,
        wakewordSensitivities: list,
        vadPath: str,
        vadThreshold: float,
        voskPath: str,
        savePath: str,
        callback: Callable,
        loop: asyncio.BaseEventLoop,
        offline: bool = False,
        device: int = None,
        **kwargs,
    ) -> None:

        # Class parameters
        self.wakewords = wakewords
        self.offline = offline
        self.savePath = savePath
        self.voskPath = voskPath
        self.device = device
        self.loop = loop
        self.sensitivities = wakewordSensitivities
        self._callback = callback

        # Class kwarg parameters
        self.speechLengths = kwargs.get("speechLengths", (6.0, 0.9))
        self.speechLengthMultiplier = kwargs.get("speechLengthMultiplier", 0.15)
        self.beforeWokeBufferLimit = kwargs.get("beforeWokeBufferLimit", 200)
        self.googleRecognizerKey = kwargs.get("googleRecognizerKey", None)
        self.disableVosk = kwargs.get("disableVosk", False)

        # Empty string convert to None
        if self.googleRecognizerKey == "":
            self.googleRecognizerKey = None

        # Initialize vosk recognizer

        if not self.disableVosk:
            self.voskModel = vosk.Model(self.voskPath)
        self.vosk = None
        self.restartVosk()

        # Initialize speechrecognition module
        self.srRecognizer = Recognizer()

        # Initialize other libraries
        w = [x for x in self.wakewords if x in pvporcupine.KEYWORDS]
        self.porcupine = None

        if w:
            self.porcupine = pvporcupine.create(
                keywords=w, sensitivities=self.sensitivities
            )

        self.vad = VAD(vadPath, vadThreshold)

        self.done = False
        self.listen = True
        self.woke = False

        self._speechLength = self.speechLengths[0]
        self._frames = {"beforeWoke": [], "afterWoke": []}
        self._followup = False
        self._q = Queue()
        self._ready = False
        self._speech = True
        self._startSpeechLength = self.speechLengths[0]
        self._realSpeechLength = self.speechLengths[1]
        self._lastRecognizedTime = time.time()

        self.__count = 0
        self.__prevSpeaking = None
        self.__length = 0

        # User callback parameters
        self.callbackParams = {}

    def __callback(self, data, frames, time_, status) -> None:
        self._q.put(bytes(data))

    def _reset(self) -> None:
        self._frames = {"beforeWoke": [], "afterWoke": []}
        
        if not self.disableVosk:
            self.vosk.FinalResult()
            
        self.woke = False
        self._speech = True
        self._lastRecognizedTime = time.time()
        self.__count = 0
        self.__prevSpeaking = None
        self.__length = 0
        self._speechLength = self.speechLengths[0]

    def multiplySpeechLength(self, multiplier: float) -> float:

        """
        Dynamically update the speech length by multiplying it by a certain value.
        """

        self._realSpeechLength = self.speechLengths[1] * multiplier
        return self._realSpeechLength

    def recognizeDone(self) -> None:

        """
        Tells the recognizer that we are done recognizing.
        """

        self._speech = False

    def restartVosk(self) -> None:
        """
        Restart just the Vosk recognizer.
        """

        if not self.disableVosk:
            self.vosk = vosk.KaldiRecognizer(self.voskModel, RATE)

    async def recognize(self) -> dict:

        if not self._speech:

            if self.offline:
                if not self.disableVosk:
                    text = json.loads(self.vosk.FinalResult())["text"]
                    return {"status": "recognized", "msg": text}
                return {"status": "error", "msg": f"both disableVosk and offline is True. Can't recognize with nothing to recognize with."}

            frames = self._frames["beforeWoke"][-10:] + self._frames["afterWoke"]

            # First save the data gathered into a .wav file
            wf = wave.open(self.savePath, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
            wf.close()

            # Convert it into a AudioData object
            try:
                with AudioFile(self.savePath) as src:
                    audio = self.srRecognizer.record(src)
            except Exception as e:
                return {
                    "status": "error",
                    "msg": f"Failed to convert cache file to AudioData. ({e})",
                }

            # Finally attempt to recognize using google's recognizer from speechrecognition module
            try:
                content = self.srRecognizer.recognize_google(
                    audio, key=self.googleRecognizerKey
                )
                callback = {"status": "recognized", "msg": content}
            except UnknownValueError:
                callback = {"status": "unknown", "msg": "Unknown value."}
            except Exception as e:
                callback = {
                    "status": "error",
                    "msg": f"Failed to recognize audio. ({e})",
                }
            finally:
                return callback

        return {"status": "listening", "msg": "Appending frames."}

    async def callback(self, *args, **kwargs) -> None:
        await self._callback(*args, **kwargs, **self.callbackParams)

    async def wakeUp(
        self, followup: bool = False, emitCallback: bool = True, **kwargs
    ) -> None:

        """
        Wake up the speech recognizer,

        Parameters
        ----------
        `followup` : bool

        """

        self.woke = True
        self._followup = followup
        self.__prevSpeaking = time.time()

        self.callbackParams = {"followup": followup, **kwargs}
        if emitCallback:
            await self.callback({"status": "woke", "msg": "woke"})

    async def start(self, blocking: bool = False) -> None:

        """
        Start the speech recognizer.

        Parameters
        ----------
        `blocking` : bool
            if True, speech recognizer will block the program.
        """

        if blocking:
            return await self._start()

        def f():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._start())

        osInterface.thread(f)
        while not self._ready:
            await asyncio.sleep(0.05)

    async def wokeListen(self, data) -> bool:

        """
        Starts listening for the provided wake words both using pvporcupine and vosk.
        Vosk will not be used if self.disableVosk is True
        """

        if not self.disableVosk:
            # Get vosk information
            self.vosk.AcceptWaveform(data)
            partial = json.loads(self.vosk.PartialResult())
        else:
            partial = {"partial": ""}

        # Get pvporcupine wake word information
        p = -1
        if self.porcupine:
            p = self.porcupine.process(np.frombuffer(data, dtype=np.int16))

        # Check if a wake word is recognized using both vosk and porcupine if porcupine is successfully initialized
        if any(k in partial["partial"] for k in self.wakewords) or p >= 0:

            if not self.disableVosk:
                self.vosk.FinalResult()

            return True
        
        # Constantly collect before wake word frames
        if len(self._frames["beforeWoke"]) > self.beforeWokeBufferLimit:
            self._frames["beforeWoke"].pop(0)
        self._frames["beforeWoke"].append(data)
        

        if not self.disableVosk:
            # Prevent active listening from getting way too big, will cause a memory leak if not implemented
            if len(partial["partial"].split()) > 25:
                self.vosk.FinalResult()
                self.restartVosk()

        vad = await self.vad.isSpeech(data)
        if vad:
            self.__prevSpeaking = time.time()

        if not self.__prevSpeaking:
            self.__prevSpeaking = time.time()
        length = time.time() - self.__prevSpeaking

        if length > 20.0:

            if not self.disableVosk:
                self.vosk.FinalResult()
                self.restartVosk()
            self.__prevSpeaking = time.time()

        # Emit what the vosk recognizer is currently hearing
        await self.callback(
            {"status": "activeListeningPartial", "msg": partial["partial"]}
        )

        return False

    async def _start(self) -> None:
        with sd.RawInputStream(
            samplerate=RATE,
            blocksize=CHUNK,
            device=self.device,
            dtype="int16",
            channels=CHANNELS,
            callback=self.__callback,
        ):
            self._ready = True
            while not self.done:
                data = self._q.get()

                if self.listen:

                    # Wait for one of the wake words to be triggered
                    if not self.woke:

                        # There seems to be a bug wherein woke becomes True right after the speech is recognized, so we do a time check to prevent that. (pls fix) FIXME
                        woke = await self.wokeListen(data)
                        if (time.time() - self._lastRecognizedTime) < 1.8:
                            woke = False

                    # Now wake up the processor/recognizer
                    if woke and not self.woke:
                        await self.wakeUp()

                    if self.woke:
                        
                        partial = None
                        if not self.disableVosk:
                            # Give vosk the speech data
                            self.vosk.AcceptWaveform(data)

                            # Realtime Partial data
                            partial = list(json.loads(self.vosk.PartialResult()).items())[
                                0
                            ][1].strip()
                            if partial:
                                await self.callback(
                                    {"status": "recognizedPartial", "msg": partial}
                                )

                        # Perform voice activity detection
                        vad = await self.vad.isSpeech(data)
                        if vad:
                            self.__count += 1
                            self.__prevSpeaking = time.time()

                            await self.callback(
                                {"status": "voiceActivity", "msg": "voiceActivity"}
                            )

                        # Perform previous voice activity checking.
                        if self.__prevSpeaking:
                            self.__length = time.time() - self.__prevSpeaking

                            comparator = self.__count == 0 or not partial
                            if self.disableVosk:
                                comparator = self.__count == 0

                            if comparator:
                                self._speechLength = self._startSpeechLength
                            else:
                                self._speechLength = self._realSpeechLength

                        # Current speech length has exceeded the provided speech length meaning we're done listening.
                        if self.__length > self._speechLength:
                            self.recognizeDone()

                        self._frames["afterWoke"].append(data)
                        recognized = await self.recognize()
                        await self.callback(recognized)

                        # Finally reset all the variables back to their default so that it can be ready for the next time the listener gets woke.
                        if not self._speech:
                            self._reset()


async def callback(data, *args, **kwargs) -> None:
    status = data.get("status", "listening")
    if status == "recognizedPartial":
        print(f"> {data['msg']}         {recognizer._realSpeechLength}", end="\r")

        if data["msg"].startswith("turn the lights off"):
            recognizer.recognizeDone()

        if data["msg"].endswith(("to", "two", "of", "and", "for")):
            recognizer.multiplySpeechLength(2.8)
        else:
            recognizer.multiplySpeechLength(1)

    if status == "recognized":
        print(f"You: {data['msg']}")

    if status == "woke":
        print(f"\nI'm listening...")

    if status == "activeListeningPartial":
        print(f"Active: {data['msg']}", end="\r")


async def main(loop: asyncio.BaseEventLoop) -> None:

    global recognizer
    recognizer = SpeechRecognizer(
        ["jarvis"],
        [1.0],
        osInterface.joinPath("models/vad.h5"),
        0.9,
        osInterface.joinPath("models/vosk"),
        osInterface.joinPath(".tmp/cache.wav"),
        callback,
        loop,
        speechLengths=(5.0, 1.2),
        offline=False,
        disableVosk=True
    )

    await recognizer.start(blocking=True)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(main(loop))
    except KeyboardInterrupt:
        loop.stop()

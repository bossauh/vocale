import pyttsx3
import sox
import sounddevice as sd
import soundfile as sf
import logging

logging.getLogger("sox").setLevel(logging.ERROR)

class Synthesizer:

    """
    A speech synthesizer that combines pyttsx3 with pysox to give a more sci-fi-ish sounding voice.

    Parameters
    ----------
    `rawPath` : str
        The path to save the raw pyttsx3 output.
    `processedPath` : str
        The path to save the processed output from pyttsx3.

    Methods
    -------
    `say(text: str)` :
        Say something.
    """

    def __init__(self, rawPath: str, processedPath: str):
        self.rawPath = rawPath
        self.processedPath = processedPath

        # Initialize pyttsx3 and set properties
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 180)

        # Initialize sox transformer and also add the effects
        self.tfm = sox.Transformer()
        self._init_effects()

    def _init_effects(self, vol: float = 1):

        """
        Can safely be modified to match the effects you want.
        Check https://pysox.readthedocs.io/en/latest/api.html
        """

        self.tfm.highpass(90)
        self.tfm.vol(vol)
        self.tfm.reverb(15)

    async def _saveRaw(self, text: str):
        self.engine.save_to_file(text, self.rawPath)
        self.engine.runAndWait()

        return self.rawPath

    async def _buildTfm(self, path1: str, path2: str):
        self.tfm.build(path1, path2)
        return path2

    async def play(self, path: str):
        data, rs = sf.read(path, dtype="float32")

        sd.play(data, rs, blocking=True)

    async def say(self, text: str):
        path = await self._saveRaw(text)
        processedPath = await self._buildTfm(path, self.processedPath)

        await self.play(processedPath)


if __name__ == "__main__":
    import asyncio

    async def main():
        synthesizer = Synthesizer("synthesized.mp3", "synthesized.wav")
        await synthesizer.say(
            "Are you sure that's what you want the new name to be? I'm glad you've actually made up your mind now."
        )

    asyncio.run(main())

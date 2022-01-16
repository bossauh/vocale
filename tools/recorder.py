import pyaudio
import click
import time
import soundfile as sf
import numpy as np
import os

from pathlib import Path
from constants import *

p = pyaudio.PyAudio()
input_stream = p.open(
    rate=RATE,
    frames_per_buffer=CHUNK,
    channels=CHANNELS,
    input=True,
    format=pyaudio.paInt16,
    input_device_index=3,
)


@click.command()
@click.option(
    "--iters",
    default=500,
    help="Number of iterations to take.",
    type=int,
    show_default=True,
)
@click.option("--label", help="The label of the recording.", required=True)
def main(iters, label) -> None:

    for i in range(3, 0, -1):
        click.echo(f"Starting in {i}...")
        time.sleep(1)

    click.echo(f"Recording for {iters} iterations...")
    recorded = []

    i += 1
    while True:
        try:
            stream = input_stream.read(CHUNK, exception_on_overflow=False)
            recorded.append(stream)

            if i > iters:
                break

            i += 1
        except KeyboardInterrupt:
            click.echo("Stopping...")
            quit()

    label_folder = os.path.join(RECORDINGS_FOLDER, label)
    Path(label_folder).mkdir(exist_ok=True, parents=True)

    path = os.path.join(label_folder, f"{int(time.time())}.wav")
    frames = b"".join(recorded)
    frames = np.frombuffer(frames, dtype=np.int16)

    with sf.SoundFile(path, "w", samplerate=RATE, channels=CHANNELS) as f:
        f.write(frames)


if __name__ == "__main__":
    main()

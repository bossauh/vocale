import numpy as np
import os
import soundfile as sf
from pathlib import Path
from constants import *

def get_dbs(stream):
    d = np.fromstring(stream, dtype=np.int16)
    db = 20 * np.log10(np.abs(np.fft.rfft(d[:2048])))
    return db

arrays = {}

for label in os.listdir(RECORDINGS_FOLDER):
    label_path = os.path.join(RECORDINGS_FOLDER, label)
    buffer = []
    frames = []

    for wav in os.listdir(label_path):
        wav_path = os.path.join(label_path, wav)
        with sf.SoundFile(wav_path) as f:
            while f.tell() < f.frames:

                # Get the decibel data
                pos = f.tell()
                data = f.buffer_read(CHUNK, dtype="int16")
                data = bytes(data)
                dbs = get_dbs(data)

                end = False
                try:
                    f.seek(pos + CHUNK)
                except RuntimeError:
                    end = True

                # Get the relevant frequencies from the list of decibels we've collected
                features = list(np.round(dbs[3:MAX_FREQ], 2))
                buffer.append(features)

                if len(buffer) == int(RATE / CHUNK * DURATION):
                    total = [x for y in buffer for x in y]
                    frames.append(total)
                    buffer.clear()
                
                if end:
                    break

    arrays[label] = np.array(frames)

Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

for k, v in arrays.items():
    path = os.path.join(DATA_PATH, f"{k}{FORMAT}")
    np.save(path, v)

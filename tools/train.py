import os
import random
import sys

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from constants import *

training = []
labels = [x[:-4] for x in os.listdir(DATA_PATH) if x.endswith(FORMAT)]


if len(labels) < 2:
    print("There is less than 2 training data. Exiting...")
    sys.exit()


for name in os.listdir(DATA_PATH):
    if name.endswith(FORMAT):
        path = os.path.join(DATA_PATH, name)
        data = np.load(path).tolist()
        y = [0] * len(labels)

        # Create the training data
        for aud in data:
            y[labels.index(name[:-4])] = 1
            training.append([aud, y])

model = Sequential()
random.shuffle(training)
training = np.array(training)
train_x, train_y = list(training[:, 0]), list(training[:, 1])

# Layers
model.add(Dense(128, activation="relu", input_shape=(len(train_x[0]),)))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(len(train_y[0]), activation="softmax"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(np.array(train_x), np.array(train_y), epochs=EPOCHS, batch_size=BATCH_SIZE)

model.save(MODEL_PATH)

import csv

import pandas as pd
import tensorflow as tf
import keras
from keras import layers
import numpy as np

pd.set_option('display.max_columns', None)

df = pd.read_csv("./Telco-Customer-Churn.csv")
df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
df["InternetService"] = df["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2})
df["Contract"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})
df["PaymentMethod"] = df["PaymentMethod"].map(
    {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3})
# https://stackoverflow.com/a/78066237
with pd.option_context("future.no_silent_downcasting", True):
    df = df.replace({"No": 0, "Yes": 1, "No internet service": 2, "No phone service": 2}).infer_objects(copy=False)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda loc: pd.to_numeric(loc, errors="coerce"))
    df = df.fillna(0).infer_objects(copy=False)

dataset = df.iloc[:, 1:].to_numpy()
np.random.shuffle(dataset)
data = dataset[:, :-1]
target = dataset[:, -1]

# Convert to one-hot encoded
target = keras.utils.to_categorical(target, 2)

split = int(len(dataset) * 0.8)

train, target_train = data[:split], target[:split]
test, target_test = data[split:], target[split:]

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_dim=19),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='softmax'),
])

model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

save_csv_path = "./model_metrics.csv"


class DataSaver(keras.callbacks.Callback):
    def __init__(self, save_path):
        super(DataSaver, self).__init__()
        self.save_path = save_path
        self.csvfile = open(save_path, 'w', newline='')
        header = ["Epoch", "Train accuracy", "Test accuracy", "Train loss", "Test loss"]
        self.writer = csv.DictWriter(self.csvfile, fieldnames=header)
        self.writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        self.writer.writerow({
            "Epoch": epoch + 1,
            "Train accuracy": logs["accuracy"],
            "Test accuracy": logs["val_accuracy"],
            "Train loss": logs["loss"],
            "Test loss": logs["val_loss"]
        })

    def on_train_end(self, logs=None):
        self.csvfile.close()


saver = DataSaver(save_csv_path)

model.fit(train, target_train, epochs=200, batch_size=64, validation_data=(test, target_test), callbacks=[saver])

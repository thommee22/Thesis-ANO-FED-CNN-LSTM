# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:37:22 2024

@author: Administrator
"""

import flwr as fl
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization,GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.metrics import Recall
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score,accuracy_score
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras import regularizers
import os
import sklearn.metrics
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import keras as ks
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import keras_cv
import math

class TestClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, val_dataset,test_dataset):
        self.model = model
        self.train = train_dataset
        self.val = val_dataset
        self.test = test_dataset
        self.history = []  
        self.round_counter = 0
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.round_counter += 1
        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.train, epochs=30, validation_data=val_dataset,callbacks=[early_stopping]
        )
        self.history.append(history.history)
        # Return updated model parameters and validation results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.train)
        results = {
            "loss": history.history["loss"][0],
            "binary_accuracy": history.history["binary_accuracy"][0],
            "recall": history.history["recall"][0],
            "precision": history.history["precision"][0],
            "val_loss": history.history["val_loss"][0],
            "val_binary_accuracy": history.history["val_binary_accuracy"][0],
            "val_recall": history.history["val_recall"][0],
            "val_precision": history.history["val_precision"][0],
        }
        self.plot_and_save_history()
	

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        num_examples_test = len(self.test)
        # Get config values

        # Evaluate global model parameters on the local test data and return results
        loss, binary_accuracy,precision,recall = self.model.evaluate(test_dataset)
        output_dict = {
                "binary_accuracy": binary_accuracy,  # accuracy from tensorflow model.evaluate
                "recall": recall,
                "precision": precision,
                        }
        return loss, num_examples_test, output_dict

    def plot_and_save_history(self):
        num = self.round_counter

        # Set up the plot layout with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 plot layout
        # Plot training and validation loss
        axs[0, 0].plot(self.history[-1]["loss"], label="Training Loss")
        axs[0, 0].plot(self.history[-1]["val_loss"], label="Validation Loss")
        axs[0, 0].set_title(f"Loss during Round {num}")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].legend()

        # Plot training and validation accuracy
        axs[0, 1].plot(self.history[-1]['binary_accuracy'], label="Training Accuracy")
        axs[0, 1].plot(self.history[-1]["val_binary_accuracy"], label="Validation Accuracy")
        axs[0, 1].set_title(f"Accuracy during Round {num}")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Accuracy")
        axs[0, 1].legend()

        # Plot training and validation recall
        axs[1, 0].plot(self.history[-1]["recall"], label="Training Recall")
        axs[1, 0].plot(self.history[-1]["val_recall"], label="Validation Recall")
        axs[1, 0].set_title(f"Recall during Round {num}")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("Recall")
        axs[1, 0].legend()

        # Plot training and validation recall
        axs[1, 1].plot(self.history[-1]["precision"], label="Training precision")
        axs[1, 1].plot(self.history[-1]["val_precision"], label="Validation precision")
        axs[1, 1].set_title(f"Precision during Round {num}")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Precision")
        axs[1, 1].legend()

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"training_history_round{num}.png")
        plt.close()

    def plot_and_save_history2(self):
        num = self.round_counter
        # Plot training history
        plt.plot(self.history[-1]["loss"], label="Training Loss")
        plt.plot(self.history[-1]["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training History Round{num}")
        plt.legend()
        # Save the plot
        plt.savefig(f"training_history_round{num}.png")
        plt.close()



# Function to create a model
def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.LSTM(128, activation='relu'),  # Optional dense layer after flattening
        tf.keras.layers.Dropout(0.5),  # Optional dropout layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss=tf.keras.losses.BinaryFocalCrossentropy(alpha=0.8, gamma=2), 
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
# Create a learning rate schedule with exponential decay
def lr_scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * math.exp(-0.1)

lr_callback = LearningRateScheduler(lr_scheduler)


def create_tf_dataset(features, labels, time_steps=100, batch_size=32, shuffle=False):
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=features,
        targets=labels,
        sequence_length=time_steps,
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=batch_size,
    )
    return dataset


def split_data_chronologically(data, test_ratio=0.2, val_ratio=0.2, target_column='anomaly_label'):
    total_samples = len(data)
    test_split_idx = int(total_samples * (1 - test_ratio))
    val_split_idx = int(test_split_idx * (1 - val_ratio / (1 - test_ratio)))

    train_data = data.iloc[:val_split_idx]
    val_data = data.iloc[val_split_idx:test_split_idx]
    test_data = data.iloc[test_split_idx:]

    train_labels = train_data[target_column].values
    val_labels = val_data[target_column].values
    test_labels = test_data[target_column].values

    train_data =train_data.drop(columns=[target_column])
    test_data =test_data.drop(columns=[target_column])
    val_data =val_data.drop(columns=[target_column])

    return train_data, val_data, test_data, train_labels, test_labels, val_labels


dataset = pd.read_csv('4135.csv', index_col='ts', parse_dates=True)

train_data, val_data, test_data, train_labels, test_labels, val_labels = split_data_chronologically(dataset)

train_data = train_data.to_numpy()
val_data = val_data.to_numpy()
test_data = test_data.to_numpy()

scaler = StandardScaler()
# Fit on training data
scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
val_data = scaler.transform(val_data)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)
val_data = tf.cast(val_data, tf.float32)

# Create datasets
train_dataset = create_tf_dataset(train_data,train_labels,time_steps=100, batch_size=128)
val_dataset = create_tf_dataset(val_data,val_labels,time_steps=100,batch_size=128)
test_dataset = create_tf_dataset(test_data,test_labels,time_steps=100,batch_size=128)

num_features = train_data.shape[-1]
timesteps= 100

# Initialize the model with the appropriate input shape
model = create_model(input_shape=(timesteps, num_features))

# Instantiate the TestClient with the model and datasets
fl.client.start_client(
    server_address="10.132.12.45:8080", client=TestClient(model, train_dataset, val_dataset, test_dataset).to_client()
)

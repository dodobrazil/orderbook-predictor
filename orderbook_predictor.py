# -*- coding: utf-8 -*-
"""
Real-Time Binance Order Book Predictor using an LSTM.

This script connects to the Binance WebSocket for BTC/USDT order book updates,
trains a small LSTM model on the initial data stream, and then provides
real-time predictions on whether the next price tick will be up or down.

Version 10: Final fix based on received key structure.
"""
import os
import sys
import json
import asyncio
import websockets
import numpy as np
import tensorflow as tf
from collections import deque
from asyncio import WindowsSelectorEventLoopPolicy

# --- Configuration ---
SYMBOL = "btcusdt"
# The WebSocket stream provides updates for the top 20 levels every 100ms
STREAM_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@depth20@100ms"

# LSTM Model Configuration
SEQUENCE_LENGTH = 20  # Number of time steps in each input sequence
# Reduced for faster testing. Set back to 2000+ for a more meaningful model.
TRAIN_SNAPSHOTS = 200 
MODEL_FILENAME = "lstm_model.keras" # Where to save/load the trained model

# --- TensorFlow Logging Control ---
# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class RealTimePredictor:
    """
    Handles data processing, model training, and real-time prediction.
    """
    def __init__(self):
        self.snapshots_for_training = []
        self.live_sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.model = None
        self.is_trained = False
        self._load_model()

    def _load_model(self):
        """Loads the pre-trained model if it exists."""
        if os.path.exists(MODEL_FILENAME):
            print(f"Found and loading existing model: {MODEL_FILENAME}")
            self.model = tf.keras.models.load_model(MODEL_FILENAME)
            self.is_trained = True
            print("Model loaded successfully. Starting prediction.")
        else:
            print(f"No pre-trained model found. Will start collecting data for training.")

    def _create_sequences(self, data):
        """
        Transforms a list of snapshots into sequences for LSTM training.
        """
        X, y = [], []
        for i in range(len(data) - SEQUENCE_LENGTH):
            sequence = data[i:(i + SEQUENCE_LENGTH)]
            X.append([item[0] for item in sequence])
            current_best_bid = sequence[-1][1]
            next_best_bid = data[i + SEQUENCE_LENGTH][1]
            y.append(1 if next_best_bid > current_best_bid else 0)
        return np.array(X), np.array(y)

    def _build_and_train(self):
        """Builds, trains, and saves the LSTM model."""
        if len(self.snapshots_for_training) < TRAIN_SNAPSHOTS:
            return

        print("\n--- Starting Model Training ---")
        print(f"Preparing {len(self.snapshots_for_training)} snapshots for training...")
        X_train, y_train = self._create_sequences(self.snapshots_for_training)
        
        if len(X_train) == 0:
            print("Not enough data to create sequences. Need more snapshots.")
            self.snapshots_for_training.clear()
            return
            
        print(f"Created {len(X_train)} sequences. Building LSTM model...")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, X_train.shape[2])),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Training model... (This may take a minute)")
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        model.save(MODEL_FILENAME)
        print(f"Model trained and saved as {MODEL_FILENAME}")
        
        self.model = model
        self.is_trained = True
        self.snapshots_for_training.clear()
        print("\n--- Training Complete. Switching to Real-Time Prediction ---")

    async def process_message(self, message):
        """Processes each incoming WebSocket message."""
        try:
            data = json.loads(message)

            # --- FINAL FIX ---
            # Use the correct keys 'bids' and 'asks' as discovered from debugging.
            if 'bids' not in data or 'asks' not in data or not data['bids'] or not data['asks']:
                return

            best_bid = float(data['bids'][0][0])
            best_ask = float(data['asks'][0][0])
            spread = best_ask - best_bid
            feature_vector = [best_bid, best_ask, spread]

            if not self.is_trained:
                self.snapshots_for_training.append((feature_vector, best_bid))
                progress = len(self.snapshots_for_training)

                # Clean progress bar
                if progress % 20 == 0 or progress == TRAIN_SNAPSHOTS:
                    print(f"\rCollecting data for training: {progress}/{TRAIN_SNAPSHOTS}", end="", flush=True)

                if progress >= TRAIN_SNAPSHOTS:
                    print() 
                    self._build_and_train()
            else:
                self.live_sequence.append(feature_vector)
                if len(self.live_sequence) == SEQUENCE_LENGTH:
                    input_data = np.array(self.live_sequence).reshape(1, SEQUENCE_LENGTH, 3)
                    prediction_prob = self.model.predict(input_data, verbose=0)[0][0]
                    prediction_symbol = "↑" if prediction_prob > 0.5 else "↓"
                    print(f"\rPrediction: {prediction_symbol} (Prob: {prediction_prob:.2f})  ", end="", flush=True)

        except Exception as e:
            print(f"\nAn error occurred during message processing: {e}")

async def main():
    """Main function to connect to WebSocket and run the predictor with auto-reconnect."""
    predictor = RealTimePredictor()
    while True:
        try:
            print("Connecting to Binance WebSocket...")
            async with websockets.connect(STREAM_URL) as websocket:
                print(f"Successfully connected to {STREAM_URL}. Now collecting data...")
                if predictor.is_trained:
                    predictor.live_sequence.clear()
                    print("Live data sequence cleared for fresh predictions.")
                
                async for message in websocket:
                    await predictor.process_message(message)

        except (websockets.exceptions.ConnectionClosedError, ConnectionResetError) as e:
            print(f"\nConnection lost: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}. Reconnecting in 10 seconds...")
            await asyncio.sleep(10)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting.")

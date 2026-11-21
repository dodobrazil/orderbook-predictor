# Order Book Predictor

Connects to the Binance WebSocket depth stream for BTC/USDT, builds features from the top 20 order book levels, trains a small LSTM model on the initial data stream, and then predicts in real time whether the next price tick will move up or down.

## Features

- Streams 20-level depth snapshots every 100 ms
- Sequence-based LSTM model (TensorFlow / Keras), saved and reloaded between runs
- Live prediction loop after the initial training phase
- `tools/ws_debug.py`: minimal WebSocket client for diagnosing connectivity issues

## Requirements

```bash
pip install tensorflow numpy websockets
```

## Usage

```bash
python orderbook_predictor.py
```

Configuration such as the symbol, sequence length and number of training snapshots is defined at the top of the script.

## Disclaimer

This is an experimental project for learning purposes and is not financial advice.

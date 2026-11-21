# -*- coding: utf-8 -*-
"""
A simple diagnostic script to test the WebSocket connection to Binance.

This script only connects to the specified stream and prints every
message it receives. It has no other logic. This helps to determine
if the network connection itself is the source of the problem.
"""
import asyncio
import websockets
import sys
from asyncio import WindowsSelectorEventLoopPolicy

# Use the exact same stream URL as the main script for a fair test
STREAM_URL = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"

async def debug_connection():
    """Connects and prints all incoming messages."""
    print("Attempting to connect to WebSocket...")
    try:
        async with websockets.connect(STREAM_URL) as websocket:
            print("--- Successfully Connected! ---")
            print("Now waiting for messages. If you see a continuous stream of text below, the connection is working.")
            print("If nothing appears after this line, the connection is being blocked.")
            
            async for message in websocket:
                # If this line is reached, it means we are receiving data
                print(message)

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(debug_connection())
    except KeyboardInterrupt:
        print("\nScript stopped by user.")


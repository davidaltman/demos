"""
MQTT Image Transfer Script

This script demonstrates a simple image transfer system using MQTT protocol.
It consists of two main components:
1. A broker (sender) that compresses and sends an image.
2. A client (receiver) that receives, decompresses, and saves the image.

The script uses PNG compression to maintain image quality and support transparency.
It also implements a basic acknowledgment mechanism to ensure successful transfer.

Requirements:
- paho-mqtt
- Pillow (PIL)

Usage:
1. Ensure 'headshot.png' is in the same directory as this script or update the image_path.
2. Run the script: python mqtt_image_transfer.py
"""

import paho.mqtt.client as mqtt
from PIL import Image
import io
import base64
import time
import multiprocessing

# MQTT settings
broker_address = "localhost"
port = 1883
topic = "image/transfer"
ack_topic = "image/ack"

# Image path
image_path = "liquid_drop.png"

# Compression level (0-9, where 0 is no compression and 9 is maximum compression)
compression_level = 6

def compress_image(image_path: str, compression_level: int) -> bytes:
    """
    Compress the image using PNG format.

    Args:
    image_path (str): Path to the input image file.
    compression_level (int): PNG compression level (0-9).

    Returns:
    bytes: Compressed image data.
    """
    with Image.open(image_path) as img:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG', compress_level=compression_level)
        return img_byte_arr.getvalue()

def run_broker():
    """
    Run the broker (sender) part of the MQTT transfer.

    This function compresses the image, sends it over MQTT,
    and waits for an acknowledgment from the client.
    """
    ack_received = False

    def on_connect(client, userdata, flags, rc, properties=None):
        print(f"Broker connected with result code {rc}")

    def on_message(client, userdata, msg):
        nonlocal ack_received
        if msg.topic == ack_topic:
            print("Acknowledgment received from client")
            ack_received = True

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker_address, port, 60)
    client.subscribe(ack_topic)
    client.loop_start()

    try:
        print("Compressing image...")
        compressed_image = compress_image(image_path, compression_level)

        encoded_image = base64.b64encode(compressed_image).decode('utf-8')

        print("Sending compressed image...")
        client.publish(topic, encoded_image)
        print("Compressed image sent")

        timeout = time.time() + 30  # 30 seconds timeout
        while not ack_received and time.time() < timeout:
            time.sleep(0.1)

        if ack_received:
            print("Transfer completed successfully")
        else:
            print("Transfer timed out waiting for acknowledgment")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        client.loop_stop()
        client.disconnect()

def run_client():
    """
    Run the client (receiver) part of the MQTT transfer.

    This function receives the compressed image over MQTT,
    saves it to a file, and sends an acknowledgment to the broker.
    """
    def on_connect(client, userdata, flags, rc, properties=None):
        print(f"Client connected with result code {rc}")
        client.subscribe(topic)

    def on_message(client, userdata, msg):
        if msg.topic == topic:
            print("Receiving image...")
            encoded_image = msg.payload.decode('utf-8')
            img_data = base64.b64decode(encoded_image)

            with io.BytesIO(img_data) as img_byte_arr:
                img = Image.open(img_byte_arr)
                img.save("received_image.png")
            print("Image received and saved as 'received_image.png'")

            client.publish(ack_topic, "ACK")
            print("Acknowledgment sent")

            time.sleep(1)
            client.disconnect()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker_address, port, 60)
    client.loop_forever()

if __name__ == "__main__":
    client_process = multiprocessing.Process(target=run_client)
    client_process.start()

    time.sleep(2)  # Give the client some time to connect

    run_broker()

    client_process.join()

print("Script completed")
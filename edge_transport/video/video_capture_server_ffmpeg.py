import cv2
import paho.mqtt.client as mqtt
import base64
import time
import subprocess
import numpy as np

"""
This script captures video from a MacBook Pro camera, compresses it using FFmpeg,
and sends it to a local MQTT broker.
"""

# MQTT settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

MQTT_TOPIC = "edge/video/stream"

# Video settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

class VideoStreamer:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.ffmpeg_command = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{FRAME_WIDTH}x{FRAME_HEIGHT}',
            '-r', str(FPS),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-tune', 'zerolatency',
            '-f', 'mpegts',
            '-'
        ]
        self.ffmpeg_process = subprocess.Popen(self.ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")

    def run(self):
        self.client.loop_start()
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                # Write frame to FFmpeg
                self.ffmpeg_process.stdin.write(frame.tobytes())

                # Read compressed data
                compressed_frame = self.ffmpeg_process.stdout.read(4096)  # Adjust buffer size as needed
                # In the run() method of VideoStreamer class
                if compressed_frame:
                    encoded_frame = base64.b64encode(compressed_frame).decode('utf-8')
                    result = self.client.publish(MQTT_TOPIC, encoded_frame)
                    print(f"Published frame, size: {len(encoded_frame)} bytes")
                else:
                    print("No compressed frame data")
                time.sleep(1/FPS)
                #time.sleep(1)


        except KeyboardInterrupt:
            print("Interrupted by user, shutting down...")
        finally:
            self.cap.release()
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.client.loop_stop()
            self.client.disconnect()

if __name__ == "__main__":
    streamer = VideoStreamer()
    streamer.run()
import paho.mqtt.client as mqtt
import base64
import cv2
import numpy as np
import subprocess

"""
This script subscribes to a compressed video stream from a local MQTT broker,
decompresses it using FFmpeg, and displays/saves the video.
"""

# MQTT settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

MQTT_TOPIC = "edge/video/stream"

# Output video file settings
OUTPUT_FILENAME = "received_video.mp4"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

class VideoReceiver:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)

        self.ffmpeg_command = [
            'ffmpeg',
            '-i', 'pipe:0',
            '-c:v', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-f', 'rawvideo',
            '-v', 'error',  # Only show errors
            'pipe:1'
        ]
        self.ffmpeg_process = subprocess.Popen(self.ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        client.subscribe(MQTT_TOPIC)

    def on_message(self, client, userdata, msg):
        print(f"Received message, size: {len(msg.payload)} bytes")
        compressed_frame = base64.b64decode(msg.payload)

        # Write compressed frame to FFmpeg
        self.ffmpeg_process.stdin.write(compressed_frame)
        self.ffmpeg_process.stdin.flush()

        # Read raw frame from FFmpeg
        raw_frame = self.ffmpeg_process.stdout.read(FRAME_WIDTH * FRAME_HEIGHT * 3)

        if len(raw_frame) == FRAME_WIDTH * FRAME_HEIGHT * 3:
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))

            # Write frame to video file
            self.out.write(frame)

            # Display the frame
            cv2.imshow('Received Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
        else:
            print(f"Incomplete frame received: {len(raw_frame)} bytes")

    def run(self):
        print(f"Listening for video stream. Press Ctrl+C to stop.")
        self.client.loop_forever()

    def stop(self):
        self.out.release()
        cv2.destroyAllWindows()
        self.client.loop_stop()
        self.client.disconnect()
        self.ffmpeg_process.stdin.close()
        self.ffmpeg_process.wait()

if __name__ == "__main__":
    receiver = VideoReceiver()
    try:
        receiver.run()
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")
    finally:
        receiver.stop()
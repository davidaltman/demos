import cv2
import paho.mqtt.client as mqtt
import base64
import time

"""
This script captures video from a MacBook Pro camera and sends it to a local MQTT broker.
It uses OpenCV to capture frames, compresses them as JPEG, and publishes them to an MQTT topic.
"""

# MQTT settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "edge/video/stream"

class VideoStreamer:
    """
    A class to capture video from a camera and stream it over MQTT.
    """

    def __init__(self):
        """
        Initialize the VideoStreamer with MQTT client and video capture settings.
        """
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)

        self.cap = cv2.VideoCapture(0)  # 0 is usually the built-in camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def on_connect(self, client, userdata, flags, rc):
        """
        Callback function for when the client receives a CONNACK response from the server.

        Args:
            client: The client instance for this callback
            userdata: The private user data as set in Client() or userdata_set()
            flags: Response flags sent by the broker
            rc: The connection result
        """
        print(f"Connected with result code {rc}")

    def run(self):
        """
        Main method to run the video streaming process.
        Captures frames, encodes them as JPEG, and publishes to MQTT topic.
        """
        self.client.loop_start()
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                encoded_frame = base64.b64encode(buffer).decode('utf-8')

                self.client.publish(MQTT_TOPIC, encoded_frame)

                time.sleep(1/30)  # Adjust for desired frame rate

        except KeyboardInterrupt:
            print("Interrupted by user, shutting down...")
        finally:
            self.cap.release()
            self.client.loop_stop()
            self.client.disconnect()

if __name__ == "__main__":
    streamer = VideoStreamer()
    streamer.run()
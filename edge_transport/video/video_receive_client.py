import paho.mqtt.client as mqtt
import base64
import cv2
import numpy as np

"""
This script subscribes to a video stream from a local MQTT broker, decodes the received frames,
saves them to a video file, and optionally displays them in real-time.
"""

# MQTT settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "edge/video/stream"

# Output video file settings
OUTPUT_FILENAME = "received_video.avi"
FPS = 30.0
FRAME_SIZE = (640, 480)

class VideoReceiver:
    """
    A class to receive video frames from an MQTT broker, save them to a file,
    and optionally display them in real-time.
    """

    def __init__(self):
        """
        Initialize the VideoReceiver with MQTT client and video writer settings.
        """
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, FRAME_SIZE)

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
        client.subscribe(MQTT_TOPIC)

    def on_message(self, client, userdata, msg):
        """
        Callback function for when a PUBLISH message is received from the server.

        Args:
            client: The client instance for this callback
            userdata: The private user data as set in Client() or userdata_set()
            msg: An instance of MQTTMessage. This is a class with members topic, payload, qos, retain.
        """
        # Decode the received frame
        jpg_buffer = base64.b64decode(msg.payload)
        frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Write the frame to the video file
        self.out.write(frame)

        # Optional: Display the frame (comment out if not needed)
        cv2.imshow('Received Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop()

    def run(self):
        """
        Start the MQTT client loop to receive and process video frames.
        """
        print(f"Listening for video stream. Press Ctrl+C to stop.")
        self.client.loop_forever()

    def stop(self):
        """
        Stop the video receiving process, release resources, and close connections.
        """
        self.out.release()
        cv2.destroyAllWindows()
        self.client.loop_stop()
        self.client.disconnect()

if __name__ == "__main__":
    receiver = VideoReceiver()
    try:
        receiver.run()
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")
    finally:
        receiver.stop()
# MQTT Video Streaming System

This project implements a simple video streaming system using MQTT (Message Queuing Telemetry Transport) protocol. It consists of two main components: a video capture server and a video receive client.

## Features

- Capture video from a MacBook Pro's built-in camera
- Stream video frames over MQTT
- Receive and display video frames in real-time
- Save received video to a file

## Requirements

- Python 3.6+
- OpenCV (`cv2`)
- Paho MQTT client
- NumPy
- An MQTT broker (e.g., Mosquitto)

## Installation

1. Clone this repository:
git clone https://github.com/davidaltman/video-streaming.git
cd mqtt-video-streaming

2. Install the required Python packages:
pip install opencv-python paho-mqtt numpy
or in an .venv:
pip install -r requirements.txt

3. Install an MQTT broker (if not already installed). For example, to install Mosquitto on macOS:
brew install mosquitto

## Usage

### Starting the MQTT Broker

Before running the scripts, make sure your MQTT broker is running. If you're using Mosquitto, you can start it with:
mosquitto


### Running the Video Capture Server

1. Open a terminal and navigate to the project directory.
2. Run the video capture server:
python video_capture_server.py

The server will start capturing video from your MacBook Pro's camera and publishing it to the MQTT broker.

### Running the Video Receive Client

1. Open another terminal and navigate to the project directory.
2. Run the video receive client:
python video_receive_client.py

The client will start receiving video frames, display them in a window, and save them to a file named `received_video.avi`.

## Configuration

You can modify the following settings in both scripts:

- `MQTT_BROKER`: The address of your MQTT broker (default is "localhost")
- `MQTT_PORT`: The port of your MQTT broker (default is 1883)
- `MQTT_TOPIC`: The MQTT topic for video streaming (default is "edge/video/stream")
- `FPS`: Frames per second for video capture and saving (default is 30.0)
- `FRAME_SIZE`: Size of video frames (default is 640x480)

## How It Works

### Video Capture Server

The server uses OpenCV to capture frames from the camera. Each frame is:
1. Encoded as a JPEG image
2. Converted to base64
3. Published to the MQTT broker on the specified topic

### Video Receive Client

The client subscribes to the MQTT topic and:
1. Receives the base64-encoded frames
2. Decodes them back into images
3. Displays the video stream in real-time
4. Saves the received frames to a video file

## Stopping the Scripts

- To stop the video capture server or receive client, press `Ctrl+C` in their respective terminals.
- To exit the video display window, press 'q'.

## Limitations

- This system is designed for local network use and may not perform well over the internet without additional optimizations.
- The video quality and frame rate may be limited by network bandwidth and MQTT message size limits.
- There's no built-in security or encryption for the video stream.

## Contributing

Contributions to improve the project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
3. The script will start both the broker and client processes. You should see output indicating the progress of the image transfer.

4. Once completed, you will find the transferred image saved as `received_image.png` in the same directory.

## Customization

You can adjust the following parameters in the script:

- `broker_address`: MQTT broker address (default is "localhost")
- `port`: MQTT broker port (default is 1883)
- `compression_level`: PNG compression level (0-9, where 0 is no compression and 9 is maximum compression)
- `image_path`: Path to the input image file

## How It Works

1. The script starts a client process and a broker process.
2. The broker compresses the specified image using PNG compression.
3. The compressed image is encoded and sent over MQTT.
4. The client receives the encoded image, decodes it, and saves it as a new file.
5. The client sends an acknowledgment back to the broker.
6. The broker confirms successful transfer upon receiving the acknowledgment.

## Limitations

- This script is designed for demonstration purposes and may need additional error handling and security measures for production use.
- It assumes a local MQTT broker. For remote brokers, you may need to adjust connection settings and implement proper authentication.
- Large images may exceed MQTT message size limits on some brokers. Consider implementing chunking for very large files.

## Contributing

Feel free to fork this repository and submit pull requests with any enhancements.

## License

This project is open-source and available under the MIT License.
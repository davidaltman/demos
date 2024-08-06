# Image Compression and Transfer using VAE and MQTT

This project demonstrates an image compression and transfer system using a Variational Autoencoder (VAE) for compression and MQTT for communication between a broker and a client.

## Features

- Image compression using a VAE
- MQTT-based communication for image transfer
- Multithreaded broker and client implementation
- Base64 encoding for data transfer
- Acknowledgment system for transfer confirmation

## Requirements

- Python 3.x
- PyTorch
- Pillow (PIL)
- paho-mqtt

## Installation

1. Clone this repository
2. Install the required packages:
`pip install -r requirements.txt`

## Usage

1. Ensure you have an MQTT broker running on `localhost:1883`. If not, you can install and run Mosquitto or any other MQTT broker.

2. Place your input image in the same directory as the script and update the `image_path` variable in the code.

3. Train the VAE model (not included in the current script) and save the weights as 'vae_weights.pth' in the same directory.

4. Run the script:
`python compress_vae.py`

The script will start both the broker and client processes. The broker will compress and send the image, while the client will receive, decompress, and save the image.

## How it works

1. The broker compresses the input image using a VAE.
2. The compressed data is encoded in base64 and sent via MQTT.
3. The client receives the data, decodes it, and decompresses it using the same VAE model.
4. The decompressed image is saved as 'received_liquid_drop.png'.
5. The client sends an acknowledgment back to the broker.

## Note

This is a demonstration script. In a real-world scenario, you would need to:
- Implement proper error handling and logging
- Secure the MQTT communication
- Optimize the VAE model for your specific use case
- Implement a training routine for the VAE

## License

[MIT License](https://opensource.org/licenses/MIT)

## Training:
# VAE Training Script for Image Compression

This script trains a Variational Autoencoder (VAE) for image compression using PyTorch. The VAE is designed to compress and reconstruct images, which can be used for efficient image transfer or storage.

## Features

- Implements a Convolutional VAE architecture
- Trains on a dataset of blood cell images
- Supports GPU acceleration (CUDA and MPS)
- Saves the trained model weights for later use

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Pillow (PIL)

## Installation

1. Clone this repository
2. Install the required packages:
pip install -r requirements.txt

## Dataset

The script is set up to train on a dataset of blood cell images. You can download the dataset from Kaggle:
[Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells?resource=download)

After downloading, extract the images to a folder and update the `train_data_path` variable in the script to point to your "TRAIN" folder.

## Usage

1. Ensure your dataset is properly set up and the path is correctly specified in the `train_data_path` variable.

2. Run the script:
`python train_vae.py`
This will take some time, and will try to use GPUs if available.

3. The script will automatically use GPU acceleration if available (CUDA or MPS). Otherwise, it will use CPU.

4. Training progress will be displayed in the console.

5. After training, the model weights will be saved as 'vae_weights.pth' in the same directory.

## Configuration

You can adjust the following parameters in the script:

- `input_dim`: Input image dimensions
- `latent_dim`: Dimension of the latent space
- `batch_size`: Number of images per batch
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for the Adam optimizer

## Model Architecture

The VAE consists of:
- An encoder with convolutional layers
- A latent space representation
- A decoder with transposed convolutional layers

The model is trained to minimize both reconstruction loss and KL divergence.

## Output

After training, the script saves the model weights as 'vae_weights.pth'. These weights can be loaded into the same VAE architecture for image compression and decompression tasks.

## Note

This script is designed for training purposes. For deployment in a real-world scenario, consider implementing additional features such as:
- Data augmentation
- Learning rate scheduling
- Model checkpointing
- Validation set evaluation

## License

[MIT License](https://opensource.org/licenses/MIT)
import paho.mqtt.client as mqtt
from PIL import Image
import io
import base64
import time
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# MQTT settings
broker_address = "localhost"
port = 1883
topic = "image/transfer"
ack_topic = "image/ack"

# Image path
image_path = "eosinophil_test.jpeg"
# VAE parameters
latent_dim = 128
input_dim = 256
##
input_dim = 256
latent_dim = 6144
batch_size = 32
num_epochs = 50

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def compress_image(image_path: str, model: VAE) -> bytes:
    transform = transforms.Compose([
        transforms.Resize((input_dim, input_dim)),
        transforms.ToTensor()
    ])

    with Image.open(image_path) as img:
        img_tensor = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        _, mu, _ = model(img_tensor)

    compressed_bytes = io.BytesIO()
    torch.save(mu, compressed_bytes)
    return compressed_bytes.getvalue()

def decompress_image(compressed_data: bytes, model: VAE) -> Image.Image:
    compressed_tensor = torch.load(io.BytesIO(compressed_data), map_location=torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        decompressed = model.decode(compressed_tensor)

    print(f"Decompressed tensor shape: {decompressed.shape}")
    decompressed_image = transforms.ToPILImage()(decompressed.squeeze(0).clamp(0, 1))
    return decompressed_image

def run_broker():
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
        model = VAE()
        # In a real scenario, you would load pre-trained weights here
        model.load_state_dict(torch.load('vae_weights.pth'))

        print(f"Compressing image: {image_path}")
        compressed_image = compress_image(image_path, model)

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
    def on_connect(client, userdata, flags, rc, properties=None):
        print(f"Client connected with result code {rc}")
        client.subscribe(topic)

    def on_message(client, userdata, msg):
        if msg.topic == topic:
            print("Receiving image...")
            encoded_image = msg.payload.decode('utf-8')
            compressed_data = base64.b64decode(encoded_image)

            model = VAE()
            # In a real scenario, you would load pre-trained weights here
            model.load_state_dict(torch.load('vae_weights.pth'))

            try:
                decompressed_image = decompress_image(compressed_data, model)
                decompressed_image.save("received_image.png")
                print("Image received, decompressed, and saved as 'received_image.png'")
            except Exception as e:
                print(f"Error during decompression: {str(e)}")

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
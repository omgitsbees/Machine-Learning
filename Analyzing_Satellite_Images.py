import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import rasterio

# --- Configuration ---
NUM_CLASSES = 5  # e.g., water, forest, urban, agriculture, barren
CLASS_NAMES = ['Water', 'Forest', 'Urban', 'Agriculture', 'Barren']

# --- Preprocessing for RGB ---
preprocess_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Preprocessing for Multi-Spectral (e.g., Sentinel-2 13 bands) ---
def preprocess_multispectral(image_array):
    # Normalize each band to 0-1
    image_array = image_array.astype(np.float32)
    image_array = (image_array - image_array.min(axis=(1,2), keepdims=True)) / (image_array.ptp(axis=(1,2), keepdims=True) + 1e-6)
    return torch.tensor(image_array).unsqueeze(0)  # Shape: (1, bands, H, W)

# --- Classification Model ---
class SatelliteClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
    def forward(self, x):
        return self.base(x)

# --- Segmentation Model (U-Net style, for demonstration) ---
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, num_classes, 2, stride=2)
        )
    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.decoder(x)
        return x

# --- Object Detection (using torchvision Faster R-CNN) ---
def object_detection(image):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.transforms.functional import to_tensor
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    with torch.no_grad():
        prediction = model([to_tensor(image)])[0]
    return prediction

# --- Cloud Masking (simple threshold on NDSI or brightness) ---
def cloud_mask(image_array, threshold=0.8):
    # Assume image_array shape: (bands, H, W), use band 3 (green) and band 11 (SWIR) for NDSI
    green = image_array[2]
    swir = image_array[11] if image_array.shape[0] > 11 else image_array[2]
    ndsi = (green - swir) / (green + swir + 1e-6)
    mask = ndsi > threshold
    return mask

# --- Change Detection (difference between two images) ---
def change_detection(image1, image2, threshold=0.2):
    # image1, image2: numpy arrays of same shape (bands, H, W)
    diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
    change_map = np.mean(diff, axis=0) > threshold
    return change_map

# --- Example Usage ---
if __name__ == "__main__":
    # --- Multi-spectral image loading (e.g., Sentinel-2) ---
    with rasterio.open("sentinel2_image.tif") as src:
        ms_image = src.read()  # shape: (bands, H, W)
    ms_tensor = preprocess_multispectral(ms_image)

    # --- Classification ---
    classifier = SatelliteClassifier(NUM_CLASSES)
    # classifier.load_state_dict(torch.load('classifier.pth'))  # Load your trained weights
    classifier.eval()
    pred_logits = classifier(ms_tensor[:, :3, :, :])  # Use RGB bands for classification
    pred_class = CLASS_NAMES[pred_logits.argmax().item()]
    print("Classification:", pred_class)

    # --- Segmentation ---
    unet = SimpleUNet(in_channels=ms_tensor.shape[1], num_classes=NUM_CLASSES)
    # unet.load_state_dict(torch.load('unet.pth'))
    unet.eval()
    seg_logits = unet(ms_tensor)
    seg_mask = seg_logits.argmax(dim=1).squeeze().cpu().numpy()
    print("Segmentation mask shape:", seg_mask.shape)

    # --- Object Detection (on RGB PIL image) ---
    rgb_image = Image.open("sample_satellite.jpg").convert('RGB')
    detections = object_detection(rgb_image)
    print("Object Detection:", detections)

    # --- Cloud Masking ---
    cloud = cloud_mask(ms_image)
    print("Cloud mask shape:", cloud.shape)

    # --- Change Detection ---
    # Load another image for change detection
    with rasterio.open("sentinel2_image_later.tif") as src2:
        ms_image2 = src2.read()
    change_map = change_detection(ms_image, ms_image2)
    print("Change map shape:", change_map.shape)
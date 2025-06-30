import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.models.lseg.lseg_net import LSegNet
from torchvision import transforms
from PIL import Image
import clip

labels = ["Grass", "Water", "Dirt"]

model = LSegNet(
    arch_option  = 0,
    block_depth  = 0,
    activation   = "lrelu",
    backbone     = "clip_vitl16_384",
    num_features = 256,
    aux_layers   = [],
)

checkpoint = torch.load("data/checkpoints/lseg_minimal_e200.ckpt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

image_path = "data/images/Wood/image_14.jpg"
image      = Image.open(image_path).convert("RGB")
transform  = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(image).unsqueeze(0)
model        = model.float()
input_tensor = input_tensor.float()

with torch.no_grad(): 
    text_tokens   = clip.tokenize(labels).to(input_tensor.device)
    text_features = model.clip_pretrained.encode_text(text_tokens)

    image_features   = model(input_tensor)
    image_features   = image_features / image_features.norm(dim=1, keepdim=True)
    text_features    = text_features / text_features.norm(dim=1, keepdim=True)
    logits           = model.logit_scale * torch.einsum("bchw,nc->bnhw", image_features, text_features)
    segmentation_map = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

def visualize_segmentation(image, segmentation_map, labels): 
    color_map          = np.random.randint(0, 255, (len(labels), 3), dtype=np.uint8)
    segmentation_image = color_map[segmentation_map]
    segmentation_image = cv2.resize(segmentation_image, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
    original_image     = np.array(image.convert("RGB"))

    blended = cv2.addWeighted(original_image, 0.5, segmentation_image, 0.5, 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.axis('off')
    plt.show()


visualize_segmentation(image, segmentation_map, labels)

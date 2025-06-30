import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import matplotlib.pyplot as plt


ENCODER      = "resnet34"
WEIGHTS_PATH = "../../data/checkpoints/space_net.pth"
model        =  smp.Unet(encoder_name=ENCODER, encoder_weights=None, in_channels=3, classes=1)
state_dict   = torch.load(WEIGHTS_PATH, map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=False)
model.eval()

def preprocess_image(image_path, target_size=(384, 384)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, target_size)

    img_input = img_resized.astype(np.float32) / 255.0  
    img_input = np.transpose(img_input, (2, 0, 1))  
    img_input = torch.tensor(img_input).unsqueeze(0)  

    return img, img_resized, img_input

IMAGE_PATH = "../../data/examples/maps/tower_map.jpg"
orig_img, resized_img, input_tensor = preprocess_image(IMAGE_PATH)

with torch.no_grad():
    prediction = model(input_tensor)  
    
pred_mask = prediction.squeeze().cpu().numpy()
pred_mask = 1 / (1 + np.exp(-pred_mask))
pred_mask = (pred_mask > 0.5).astype(np.uint8)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(resized_img)
plt.title("Resized Image (384×384)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Mask (384×384)")
plt.axis("off")

plt.tight_layout()
plt.show()

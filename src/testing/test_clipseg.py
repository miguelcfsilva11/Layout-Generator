import torch
from PIL import Image
import requests
from torchvision import transforms
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model     = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
image     = Image.open("data/images/Wood/image_42.jpg").convert("RGB")
prompt    = "tree"
inputs    = processor(text=prompt, images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

seg = torch.sigmoid(outputs.logits)

if len(seg.shape) == 2:   seg = seg.unsqueeze(0).unsqueeze(0)
elif len(seg.shape) == 3: seg = seg.unsqueeze(1)

seg_resized = torch.nn.functional.interpolate(
    seg,
    size=image.size[::-1],
    mode='bilinear',
    align_corners=False
).squeeze().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(image)
plt.imshow(seg_resized, cmap="viridis", alpha=0.5)
plt.title(f"CLIPSeg Output: {prompt}")
plt.axis("off")
plt.tight_layout()
plt.show()
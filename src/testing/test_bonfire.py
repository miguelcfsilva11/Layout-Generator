import torch
import clip
from PIL import Image
from src.classification.sam import SAM2Model
import numpy as np

device                  = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess       = clip.load("ViT-B/32", device=device)
image_path              = "data/images/Wood/image_15.jpg"
img                     = Image.open(image_path).convert("RGB")
x, y, w, h              = 450, 330, 30, 30
center_x                = x + w // 2
center_y                = y + h // 2

crop_width, crop_height = 128, 128
left                    = max(center_x - crop_width // 2, 0)
upper                   = max(center_y - crop_height // 2, 0)
right                   = min(center_x + crop_width // 2, img.width)
lower                   = min(center_y + crop_height // 2, img.height)
cropped_img             = img.crop((left, upper, right, lower))
sam_model               = SAM2Model(device=device)


sam_model.load_image(image_path)

clip_input = preprocess(cropped_img).unsqueeze(0).to(device)
text = clip.tokenize(["bonfire", "campfire", "flames"]).to(device)

with torch.no_grad():
    image_features  = model.encode_image(clip_input)
    text_features   = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features  /= text_features.norm(dim=-1, keepdim=True)
    similarity      = (image_features @ text_features.T).softmax(dim=-1)

    best_score, best_index = similarity[0].max(0)

clip_guided_point = (center_x, center_y)
print("Point:", clip_guided_point)
print("Score:", best_score.item())
print("Matched Prompt:", ["bonfire", "campfire", "flames"][best_index.item()])


point_coords     = np.array([[center_x, center_y]])
point_labels     = np.array([1])
masks, scores, _ = sam_model.predict(point_coords=point_coords, point_labels=point_labels)
sam_model.show_masks(masks, scores, point_coords=point_coords, input_labels=point_labels)

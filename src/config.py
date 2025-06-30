from PIL import Image, ImageTk
import torch
from transformers import CLIPProcessor, CLIPModel

textures = {
    "Forest": Image.open("data/assets/Tree.png"),
    "Water" : Image.open("data/assets/Water.png"),
    "Roof"  : Image.open("data/assets/Structure.png"),
    "Grass" : Image.open("data/assets/Grass.png"),
    "Bridge": Image.open("data/assets/Wall.png"),
    "Rock"  : Image.open("data/assets/Rock.png"),
    "House" : Image.open("data/assets/Structure.png"),
    "Castle": Image.open("data/assets/Castle.png"),
    "Dirt"  : Image.open("data/assets/Dirt.png"),
    "Object": Image.open("data/assets/Structure.png"),
    "Ruins" : Image.open("data/assets/Wall.png"),
    "Stairs": Image.open("data/assets/Structure.png"),
    "Tree"  : Image.open("data/assets/Tree.png")
}

bitmask_to_tile = {
            0 :  0,  # No neighbors: single, isolated tile
            1 :  1,  # Up only
            2 :  2,  # Right only
            3 :  3,  # Up + Right
            4 :  4,  # Down only
            5 :  5,  # Up + Down
            6 :  6,  # Right + Down
            7 :  7,  # Up + Right + Down
            8 :  8,  # Left only
            9 :  9,  # Up + Left
            10: 10, # Right + Left  
            11: 11, # Up + Right + Left
            12: 12, # Down + Left
            13: 13, # Up + Down + Left
            14: 14, # Right + Down + Left
            15: 15  # Up + Right + Down + Left (fully surrounded)
}

grid_cell_size   = 16
structure_labels = ["House", "Castle", "Object", "Stairs"]
text_labels      = ["Forest", "Water", "Castle", "Bridge", "Rock", "Dirt", "Ruins", "Grass"]
MIN_BBOX_WIDTH   = 50
MIN_BBOX_HEIGHT  = 50

device           = "cuda" if torch.cuda.is_available() else "cpu"
model            = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor        = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

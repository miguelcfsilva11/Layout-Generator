import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure, img_as_float
from src.classification.labelers.slic_labeler import SlicLabeler
from PIL import Image
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

custom_colors = {
    "Dirt" : "#8B4513",
    "Grass": "#7CFC00",
    "Water": "#1E90FF"
}

def map_labels_to_rgb(label_array, color_map): 
    height, width    = label_array.shape
    rgb_image        = np.zeros((height, width, 3), dtype=np.uint8)

    for label, hex_color in color_map.items(): 
        mask            = label_array == label
        rgb             = np.array(mcolors.to_rgb(hex_color)) * 255
        rgb_image[mask] = rgb.astype(np.uint8)

    return rgb_image

def create_legend_from_color_map(color_map):
    return [Patch(facecolor=color, edgecolor='black', label=label) for label, color in color_map.items()]

image_path        = "data/images/Wood/image_14.jpg"
image             = img_as_float(io.imread(image_path))
lab_image         = color.rgb2lab(image)
lab_clahe         = lab_image.copy()
l_channel         = lab_clahe[..., 0]
l_normalized      = l_channel / 100.0
l_eq              = exposure.equalize_adapthist(l_normalized)
lab_clahe[..., 0] = l_eq * 100.0
image_clahe       = color.lab2rgb(lab_clahe)

text_labels = list(custom_colors.keys())
n_segments  = [300]

pil_img       = Image.open(image_path).convert("RGB")
pil_img_clahe = Image.fromarray((image_clahe * 255).astype(np.uint8))

labeler_adaptive   = SlicLabeler(pil_img, n_segments=n_segments, text_labels=text_labels)
labeler_growing    = SlicLabeler(pil_img, n_segments=n_segments, text_labels=text_labels)
labeler_nongrowing = SlicLabeler(pil_img, n_segments=n_segments, text_labels=text_labels)
labeler_clahe_grow = SlicLabeler(pil_img_clahe, n_segments=n_segments, text_labels=text_labels)

labeler_growing.labelling(num_components=300, growing=True, output=False)
labeler_nongrowing.labelling(num_components=300, growing=False, output=False)
labeler_clahe_grow.labelling(num_components=300, growing=True, output=False)
labeler_adaptive.adaptive_labelling(output=False)

rgb_adaptive      = map_labels_to_rgb(labeler_adaptive.pixel_labels, custom_colors)
rgb_growing       = map_labels_to_rgb(labeler_growing.pixel_labels, custom_colors)
rgb_nongrowing    = map_labels_to_rgb(labeler_nongrowing.pixel_labels, custom_colors)
rgb_clahe_growing = map_labels_to_rgb(labeler_clahe_grow.pixel_labels, custom_colors)

legend_patches = create_legend_from_color_map(custom_colors)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(image)
axs[0, 0].set_title('Original RGB')
axs[0, 0].axis('off')

axs[0, 1].imshow(rgb_growing)
axs[0, 1].set_title('SLIC w/ Growing=True')
axs[0, 1].legend(handles=legend_patches, loc='lower left', fontsize='small', frameon=True)
axs[0, 1].axis('off')

axs[0, 2].imshow(rgb_nongrowing)
axs[0, 2].set_title('SLIC w/ Growing=False')
axs[0, 2].legend(handles=legend_patches, loc='lower left', fontsize='small', frameon=True)
axs[0, 2].axis('off')

axs[1, 0].imshow(image_clahe)
axs[1, 0].set_title('CLAHE-enhanced RGB')
axs[1, 0].axis('off')

axs[1, 1].imshow(rgb_clahe_growing)
axs[1, 1].set_title('CLAHE SLIC w/ Growing=True')
axs[1, 1].legend(handles=legend_patches, loc='lower left', fontsize='small', frameon=True)
axs[1, 1].axis('off')

axs[1, 2].imshow(rgb_adaptive)
axs[1, 2].set_title('Adaptive Labelling')
axs[1, 2].legend(handles=legend_patches, loc='lower left', fontsize='small', frameon=True)
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()

from PIL import Image
from src.classification.labelers.clipseg_labeler import CLIPSegLabeler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


dataset_path = "data/output/image_objects.csv"
df           = pd.read_csv(dataset_path)

for index, row in df.iterrows(): 
    image_path  = row['path']
    image_path  = image_path.replace("../../", "")
    objects     = eval(row['objects'])
    terrains    = eval(row['terrains'])
    text_labels = objects + terrains

    image = Image.open(image_path)

    clipseg_labeler = CLIPSegLabeler(image=image, text_labels=text_labels, object_labels=objects, terrain_labels=terrains)
    grid_labels     = clipseg_labeler.segment_and_label_grid()

    print(f"Grid Label Map for image {image_path}:")
    print(grid_labels.shape)
    
    clipseg_labeler.visualize_overlays()


    print(f"Grid Label Map for image {image_path}:")
    print(np.array(grid_labels).shape)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    num_rows , num_cols   = len(grid_labels), len(grid_labels[0])
    img_width, img_height = image.size
    cell_width            = img_width / num_cols
    cell_height           = img_height / num_rows

    unique_labels         = sorted(set(label for row in grid_labels for label in row))
    label_to_color_idx    = {label: i for i, label in enumerate(unique_labels)}
    cmap                  = plt.cm.get_cmap('tab20', len(unique_labels))

    overlay = np.zeros((num_rows, num_cols))
    for y in range(num_rows): 
        for x in range(num_cols): 
            overlay[y, x] = label_to_color_idx[grid_labels[y][x]]

    overlay_img = np.kron(overlay, np.ones((int(cell_height), int(cell_width))))
    ax.imshow(overlay_img, cmap=cmap, alpha=0.5, extent=(0, img_width, img_height, 0))
    handles = [
        mpatches.Patch(color=cmap(i), label=label)
        for label, i in label_to_color_idx.items()
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f"Label Grid Overlay (Colored): {image_path}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
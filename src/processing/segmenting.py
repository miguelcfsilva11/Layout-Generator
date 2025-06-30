import pickle
import pandas as pd
from src.classification.map_segmenter import MapSegmenter
from tqdm import tqdm 
import sys
import ast
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.colors import ListedColormap

def visualize_image_and_grid(image_path, grid_labels):
    try:

        image        = Image.open(image_path).convert("RGB")
        grid_np_str  = np.array(grid_labels)
        grid_np      = np.vectorize(lambda x: label_to_idx.get(x, 0))(grid_np_str)
        num_labels   = len(label_to_idx)
        tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        full_colors  = [tab20_colors[i % 20] for i in range(num_labels)]
        cmap         = ListedColormap(full_colors)
        fig, axs     = plt.subplots(1, 2, figsize=(14, 7))

        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        im = axs[1].imshow(grid_np, cmap=cmap, interpolation="nearest", vmin=0, vmax=num_labels - 1)
        axs[1].set_title("Segmented Grid Labels")
        axs[1].axis("off")

        unique_labels   = sorted(set(np.unique(grid_np_str)))
        legend_elements = []

        for label in unique_labels:
            idx  = label_to_idx[label]
            rgba = cmap(idx)
            legend_elements.append(Patch(facecolor=rgba, edgecolor='black', label=label))

        axs[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Failed to visualize: {image_path} | Error: {e}")


df         = pd.read_csv("data/output/image_objects.csv")
ms         = MapSegmenter()
results    = []
all_labels = set()

for row in df["terrains"].tolist() + df["objects"].tolist():
    try:
        items = ast.literal_eval(row)
        all_labels.update(item.strip() for item in items)
    except:
        continue

label_to_idx = {label: i for i, label in enumerate(sorted(all_labels))}

for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    try:

        path          = row["path"].replace("../../", "")

        if "Volcano" in path or "Snow" in path:
            continue

        objects_list  = ast.literal_eval(row["objects"])
        terrains_list = ast.literal_eval(row["terrains"])
        objects       = [obj.strip() for obj in objects_list]
        terrains      = [terrain.strip() for terrain in terrains_list]
        description   = row["description"]
        grid_labels   = ms.process_image(path, objects, terrains)   

        results.append({
            "description": description,
            "grid_labels": grid_labels,
            "path": path,
        })

        visualize_image_and_grid(path, grid_labels)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted. Saving the processed data...")    
        with open('data/output/sad_processed_data.pkl', 'wb') as f:
            pickle.dump(results, f)
        sys.exit(0)
    except Exception as e:
        print(f"Error processing image: {path}: {e}")
        continue

with open('data/output/sad_processed_data.pkl', 'wb') as f:
    pickle.dump(results, f)
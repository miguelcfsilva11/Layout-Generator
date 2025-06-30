import tkinter as tk
import pickle
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from src.config import grid_cell_size, textures, MIN_BBOX_WIDTH, MIN_BBOX_HEIGHT, structure_labels


class LabelSelectorApp:
    def __init__(self, master, image_original, overlayed_image, label_to_masks, grid_labels):
        
        self.master = master
        self.master.title("Label Selector")
        self.master.geometry("900x600")

        self.image_original  = image_original.convert("RGBA")
        self.overlayed_image = overlayed_image
        self.label_to_masks  = label_to_masks
        self.grid_labels     = grid_labels
        self.grid_image      = self.create_grid_image()
        self.view_mode       = "original"
        self.show_original   = True 

        self.canvas          = tk.Canvas(master)

        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_resize)
        
        self.base_photo      = ImageTk.PhotoImage(self.overlayed_image)
        self.canvas_image    = self.canvas.create_image(0, 0, image=self.base_photo, anchor="nw")
        self.current_photo   = self.base_photo

        self.bbox_image      = self.create_bbox_image()
        self.button_frame    = tk.Frame(master, bg="gray", height=50)
        self.button_frame.pack(side=tk.TOP)
        self.create_buttons()

        self.view_image = {
            "original": self.image_original,
            "grid": self.grid_image,
            "bbox": self.bbox_image,
        }

    def create_buttons(self):

        for label in self.label_to_masks.keys():
            btn = tk.Button(self.button_frame, text=label, command=lambda l=label: self.on_label_click(l))
            btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        toggle_btn = tk.Button(self.button_frame, text="Toggle Original/Grid", command=self.toggle_view)
        toggle_btn.pack(side=tk.LEFT, padx=5, pady=5)
        bbox_btn = tk.Button(self.button_frame, text="Bounding Box View", command=self.show_bbox_view)
        bbox_btn.pack(side=tk.LEFT, padx=5, pady=5)


    def create_grid_image(self):
        rows, cols = self.grid_labels.shape
        grid_image = Image.new("RGBA", (cols * grid_cell_size, rows * grid_cell_size))

        for i in range(rows):
            for j in range(cols):
                label = self.grid_labels[i, j]
                if label in textures:
                    resized_texture = textures[label].resize((grid_cell_size, grid_cell_size), Image.NEAREST)
                    grid_image.paste(resized_texture, (j * grid_cell_size, i * grid_cell_size), resized_texture)
                if label not in textures:
                    print(label)
        return grid_image

    def get_connected_bounding_boxes(self, mask):
        """
        Given a PIL mask image, returns a list of bounding boxes for each connected component.
        Each bounding box is a tuple (x_min, y_min, x_max, y_max).
        """
        mask_np = np.array(mask)
        if mask_np.ndim > 2:
            mask_np = mask_np[..., 0]
        binary_mask = mask_np > 0
        
        visited = np.zeros_like(binary_mask, dtype=bool)
        boxes = []
        rows, cols = binary_mask.shape
        
        for i in range(rows):
            for j in range(cols):
                if binary_mask[i, j] and not visited[i, j]:

                    queue         = [(i, j)]
                    visited[i, j] = True
                    min_i, min_j  = i, j
                    max_i, max_j  = i, j
                    
                    while queue:
                        ci, cj = queue.pop(0)
                        for ni, nj in [(ci-1, cj), (ci+1, cj), (ci, cj-1), (ci, cj+1)]:
                            if 0 <= ni < rows and 0 <= nj < cols:
                                if binary_mask[ni, nj] and not visited[ni, nj]:
                                    visited[ni, nj] = True
                                    queue.append((ni, nj))
                                    
                                    min_i = min(min_i, ni)
                                    min_j = min(min_j, nj)
                                    max_i = max(max_i, ni)
                                    max_j = max(max_j, nj)

                    boxes.append((min_j, min_i, max_j+1, max_i+1))
        return boxes

    def create_bbox_image(self):

        bbox_image = Image.new("RGBA", self.image_original.size, "white")
        draw = ImageDraw.Draw(bbox_image)

        for label, mask_data in self.label_to_masks.items():

            if label not in structure_labels:
                continue

            if isinstance(mask_data, list):
                for crop, x_min, y_min in mask_data:
                    bbox = (x_min, y_min, x_min + crop.width, y_min + crop.height)
                    draw.rectangle(bbox, outline="black", width=2)
                    draw.text((x_min, y_min), label, fill="black")

            else:
                mask  = mask_data
                boxes = self.get_connected_bounding_boxes(mask)
                for bbox in boxes:
                    if bbox[2] - bbox[0] < MIN_BBOX_WIDTH or bbox[3] - bbox[1] < MIN_BBOX_HEIGHT:
                        continue
                    draw.rectangle(bbox, outline="black", width=2)
                    draw.text((bbox[0], bbox[1]), label, fill="black")

        return bbox_image

    def toggle_view(self):

        if self.view_mode == "bbox":

            self.view_mode     = "original"
            self.show_original = True
            self.update_canvas_image(self.image_original)
        else:

            self.show_original = not self.show_original
            self.view_mode     = "original" if self.show_original else "grid"
            new_image          = self.image_original if self.show_original else self.grid_image

            self.update_canvas_image(new_image)

    def show_bbox_view(self):

        self.view_mode = "bbox"
        self.update_canvas_image(self.bbox_image)

    def update_canvas_image(self, new_image):

        self.canvas_width, self.canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        resized_image               = new_image.resize((self.canvas_width, self.canvas_height))
        new_photo                   = ImageTk.PhotoImage(resized_image)
        self.current_photo          = new_photo

        self.canvas.itemconfig(self.canvas_image, image=new_photo)

    def on_resize(self, event):
        self.update_canvas_image(self.view_image[self.view_mode])


    def on_label_click(self, label):

        canvas_width  = self.canvas.winfo_width()
        canvas_heigh  = self.canvas.winfo_height()
        scale_x       = self.canvas_width / float(self.overlayed_image.width)
        scale_y       = self.canvas_height / float(self.overlayed_image.height)

        composite = Image.new("RGBA", (self.canvas_width, self.canvas_height), (0, 0, 0, 0))

        if isinstance(self.label_to_masks[label], list):
            for crop, x_min, y_min in self.label_to_masks[label]:

                scaled_x_min = int(x_min * scale_x)
                scaled_y_min = int(y_min * scale_y)
                scaled_crop  = crop.resize((int(crop.width * scale_x), int(crop.height * scale_y)))

                composite.paste(scaled_crop, (scaled_x_min, scaled_y_min), scaled_crop)
        else:
            mask = self.label_to_masks[label]
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray(mask)
            bbox = mask.getbbox()
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox

                cropped_original = self.image_original.crop(bbox).convert("RGBA")
                cropped_mask     = mask.crop(bbox)
                scaled_x_min     = int(x_min * scale_x)
                scaled_y_min     = int(y_min * scale_y)
                scaled_width     = max(1, int(cropped_original.width * scale_x))
                scaled_height    = max(1, int(cropped_original.height * scale_y))
                scaled_crop      = cropped_original.resize((scaled_width, scaled_height))
                scaled_mask      = cropped_mask.resize((scaled_width, scaled_height))

                composite.paste(scaled_crop, (scaled_x_min, scaled_y_min), scaled_mask)

        darkened           = self.overlayed_image.resize((self.canvas_width, self.canvas_height))
        dark_overlay       = Image.new("RGBA", darkened.size, (0, 0, 0, 150))
        darkened           = Image.alpha_composite(darkened, dark_overlay)
        result             = Image.alpha_composite(darkened, composite)
        new_photo          = ImageTk.PhotoImage(result)
        self.current_photo = new_photo

        self.canvas.itemconfig(self.canvas_image, image=new_photo)


pickle_filename      = "data/output/label_to_masks.pkl"
image_original       = Image.open("data/output/map.jpg").convert("RGBA")
overlayed_image      = Image.open("data/output/map.jpg").convert("RGBA")
grid_labels_filename = "data/output/grid_labels.pkl"

with open(grid_labels_filename, "rb") as f:
    grid_labels = pickle.load(f)
    grid_labels = np.array(grid_labels)

with open(pickle_filename, "rb") as f:
    label_to_masks = pickle.load(f)

root = tk.Tk()
app = LabelSelectorApp(root, image_original, overlayed_image, label_to_masks, grid_labels)
root.mainloop()

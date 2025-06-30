import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import pickle
import random

import csv
import ast


class LabelMarker:
    """
    A tool for labeling a set of images by overlaying a grid and painting label indices.
    Each image is expected to be size 768x768, producing a 64x64 grid when using cell_size=16.
    Saves the labeled grids as .npy files for downstream use.
    """
    def __init__(self, master, image_dir, output_dir,
                 labels, cell_size=12, brush_sizes=(1,3,5,7)):
        self.master = master
        self.image_dir = image_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.image_metadata = self._load_csv_metadata("data/output/image_objects.csv")


        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, f))

        random.seed(2)
        self.image_paths = random.sample(self.image_paths, 70)
        self.idx = 0

        self.labels = labels
        self.label_to_idx = {lbl:i for i,lbl in enumerate(self.labels)}
        self.idx_to_label = {i:lbl for lbl,i in self.label_to_idx.items()}
        self.active_label = self.labels[0]

        self.cell_size = cell_size
        self.rows = self.cols = 768 // cell_size
        self.grid_data = np.full((self.rows, self.cols), fill_value=-1, dtype=int)
        self.idx_to_label[-1] = None

        self.brush_sizes = brush_sizes
        self.brush_size = brush_sizes[0]

        self.canvas = tk.Canvas(master, width=768, height=768)
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)


        ctrl = tk.Frame(master)
        ctrl.pack()

        self.label_var = tk.StringVar()
        self.label_menu = tk.OptionMenu(ctrl, self.label_var, "", command=self.set_active_label)
        self.label_menu.pack(side=tk.LEFT)


        self.label_to_idx["Eraser"] = -1
        self.idx_to_label[-1] = "Eraser"


        self.brush_var = tk.IntVar(value=self.brush_size)
        tk.OptionMenu(ctrl, self.brush_var, *self.brush_sizes, command=self.set_brush_size).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Prev", command=self.prev_image).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Next", command=self.next_image).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Save", command=self.save_labels).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Fill Empty", command=self.fill_empty_cells).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Eraser", command=lambda: self.set_active_label("Eraser")).pack(side=tk.LEFT)

        self.image_tk = None
        self.draw_objs = []  
        self.load_image()
        self.rects = {}
        
    

    def _update_label_menu(self, labels_for_image):
        menu = self.label_menu["menu"]
        menu.delete(0, "end")
        for lbl in labels_for_image:
            menu.add_command(label=lbl, command=lambda l=lbl: self.set_active_label(l))
        self.label_var.set(labels_for_image[0])
        self.active_label = labels_for_image[0]

    def _load_csv_metadata(self, csv_path):
        metadata = {}
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                path = os.path.normpath(row["path"])
                objects = ast.literal_eval(row["objects"])
                terrains = ast.literal_eval(row["terrains"])
                labels = [obj.strip() for obj in objects + terrains]
                metadata[path] = labels

        return metadata

    def fill_empty_cells(self):
        mask = self.grid_data == -1
        if not mask.any():
            print("No empty cells to fill.")
            return
        fill_idx = self.label_to_idx[self.active_label]
        self.grid_data[mask] = fill_idx

        rows, cols = np.where(mask)
        for r, c in zip(rows, cols):
            self._draw_cell_overlay(r, c)
        print(f"Filled {len(rows)} empty cells with label '{self.active_label}'.")


    def set_active_label(self, label):
        self.active_label = label
        self.label_var.set(label)  # <- this is what makes it show up in the UI

    def set_brush_size(self, size):
        self.brush_size = int(size)

    def load_image(self):
        path = self.image_paths[self.idx]

        img = Image.open(path).resize((768,768), Image.Resampling.LANCZOS)
        self.base_img = img
        self.image_tk = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0,0,anchor='nw',image=self.image_tk)


        self._draw_grid()
        self.grid_data[:] = -1

        rel_path = os.path.normpath(self.image_paths[self.idx])
        labels_for_image = self.image_metadata.get("../../" + rel_path, [])
        if not labels_for_image:
            print(f"No labels found for image: {rel_path}")
            labels_for_image = ["Unknown"]
        self._update_label_menu(labels_for_image)


    def _draw_grid(self):
        for i in range(self.rows+1):
            y = i*self.cell_size
            self.canvas.create_line(0,y,768,y,fill='black')
        for j in range(self.cols+1):
            x = j*self.cell_size
            self.canvas.create_line(x,0,x,768,fill='black')

    def on_click(self, event):
        self.paint(event)
    def on_drag(self, event):
        self.paint(event)

    def paint(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        half = self.brush_size // 2
        for dr in range(-half, half+1):
            for dc in range(-half, half+1):
                r, c = row+dr, col+dc
                if 0<=r<self.rows and 0<=c<self.cols:
                    self.grid_data[r,c] = self.label_to_idx[self.active_label]
                    self._draw_cell_overlay(r,c)

    def _draw_cell_overlay(self, row, col):

        x0, y0 = col*self.cell_size, row*self.cell_size
        x1, y1 = x0+self.cell_size, y0+self.cell_size
        if (row, col) in self.rects:
            rect = self.rects.pop((row, col))
            self.canvas.delete(rect)
        if self.active_label != "Eraser":
            color = self._color_for_label(self.active_label)
            rect = self.canvas.create_rectangle(x0,y0,x1,y1,fill=color, outline='')
            self.rects[(row, col)] = rect



    def _color_for_label(self, label):
        
        idx = self.label_to_idx[label]
        rng = np.random.RandomState(idx)
        r,g,b = rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)
        return f'#{r:02x}{g:02x}{b:02x}'

    def prev_image(self):
        if self.idx>0:
            self.idx -= 1
            self.load_image()
    def next_image(self):
        if self.idx < len(self.image_paths)-1:
            self.idx += 1
            self.load_image()

    def save_labels(self):
        str_grid = [[self.idx_to_label[i] for i in row] for row in self.grid_data]

        # Split back into subdirectory and image base name
        path = self.image_paths[self.idx]
        parent_dir = os.path.basename(os.path.dirname(path))
        base_name = os.path.splitext(os.path.basename(path))[0]

        # Create subdirectory inside output_dir
        output_subdir = os.path.join(self.output_dir, parent_dir)
        os.makedirs(output_subdir, exist_ok=True)

        # Save using base name (e.g., Image_1_labels.npy inside Wood/)
        out_path = os.path.join(output_subdir, base_name + '_labels.npy')
        np.save(out_path, str_grid)
        print(f"Saved labels to {out_path}")


if __name__ == '__main__':

    root = tk.Tk()
    root.title('Label Marker')
    image_dir  = 'data/images'
    output_dir = 'data/output/labels'
    labels     = ["House", "Fortress", "Barrel", "Bridge", "Church", "Fountain", "Tree", "Hut", "Tent", "Garden", "Windmill", "Cannon", "Well", "Lighthouse", "Farm", "Bonfire", "Table", "Boat", "Chest", "Spikes", "Statue", "Wagon", "Sign", "Lantern", "Cage", "Weapon", "Ladder", "Crate", "Cauldron", "Anvil", "Store", "Snow", "Ice", "Water", "Dirt", "Grass", "Rock", "Lava", "Swamp", "Sand"]
    app        = LabelMarker(root, image_dir, output_dir, labels)
    root.mainloop()

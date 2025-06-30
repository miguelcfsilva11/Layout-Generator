import tkinter as tk
from PIL import Image, ImageTk
from src.config import textures
from transformers import BertTokenizer
from src.models.procedural_generator import ProceduralGenerator
from src.models.components import GridVAE, expand_embedding_weights
import torch
import numpy as np
import pickle
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class CellEditor:
    
    def __init__(self, master, rows=64, cols=64, cell_size=10):

        self.master         = master
        self.rows           = rows
        self.cols           = cols
        self.cell_size      = cell_size
        
        with open("data/output/specific_processed_data.pkl", "rb") as f:
            data = pickle.load(f)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        all_labels          = {cell for item in data for row in item["grid_labels"] for cell in row if cell is not None}
        label_to_idx        = {label: i for i, label in enumerate(sorted(all_labels))}

        self.idx_to_label   = {i: label for label, i in label_to_idx.items()}
        self.label_to_idx   = label_to_idx
        self.active_label   = list(label_to_idx.keys())[0]
        self.labels         = list(label_to_idx.keys())
        self.num_labels     = len(label_to_idx)
        self.textures       = textures
        self.grid_data      = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        
        self.cell_images    = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.cell_image_ids = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        
        grid_H, grid_W                                  = 64, 64
        embed_dim, num_heads, ff_dim                    = 64, 4, 128
        time_embed_dim, diffusion_hidden_dim, timesteps = 128, 256, 1000
        batch_size, num_epochs                          = 4, 10

        num_labels_with_mask = len(label_to_idx) + 1
        vae_ckpt = "data/checkpoints/vae_best_model.pth"
        vae      = GridVAE(len(label_to_idx), embed_dim, grid_H, grid_W, latent_channels=32)
        state_dict = torch.load(vae_ckpt, map_location="cpu")  
        vae.load_state_dict(state_dict)
        vae.embedding.embedding = expand_embedding_weights(vae.embedding.embedding, len(label_to_idx) + 1)
        vae.to(device)
        vae.eval()

        model           = ProceduralGenerator(embed_dim, num_heads, ff_dim, time_embed_dim,
                                    diffusion_hidden_dim, (grid_H, grid_W), len(label_to_idx), grid_H, grid_W, 32, timesteps)
        model.vae       = vae
        self.model      = model.to(device)
        
        self.model.load_state_dict(torch.load("data/checkpoints/procedural_generator_model_epoch.pth", map_location="cpu"), strict=False)
        self.model.eval()
        self.canvas = tk.Canvas(master, width=self.cols * self.cell_size,
                                height=self.rows * self.cell_size, bg="white")
        self.canvas.pack(side=tk.TOP, padx=10, pady=10)
        self.draw_grid()
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        
        self.button_frame   = tk.Frame(master)
        self.button_frame.pack(side=tk.TOP, pady=5)
        self.label_var      = tk.StringVar(value=self.active_label)
        self.label_dropdown = tk.OptionMenu(self.button_frame, self.label_var, *self.labels, command=self.set_active_label)
        self.label_dropdown.pack(side=tk.LEFT, padx=5)

        brush_label     = tk.Label(self.button_frame, text="Brush:")
        brush_label.pack(side=tk.LEFT, padx=(10, 2))
        self.brush_size = 1 

        self.brush_var = tk.StringVar(value="1")
        brush_sizes = ["1", "3", "5", "7"]
        brush_dropdown = tk.OptionMenu(self.button_frame, self.brush_var, *brush_sizes, command=self.set_brush_size)
        brush_dropdown.pack(side=tk.LEFT)

        self.text_input = tk.Entry(self.button_frame)
        self.text_input.pack(side=tk.LEFT, padx=5)
        self.text_input.insert(0, "complete grid")


        complete_btn = tk.Button(self.button_frame, text="Complete Grid", command=self.complete_grid)
        complete_btn.pack(side=tk.LEFT, padx=5)

        reset_btn = tk.Button(self.button_frame, text="Reset Grid", command=self.reset_canvas)
        reset_btn.pack(side=tk.LEFT, padx=5)

    def set_brush_size(self, size_str):
        self.brush_size = int(size_str)
        print(f"Brush size set to: {self.brush_size}")

    def reset_canvas(self):

        for r in range(self.rows):
            for c in range(self.cols):
                self.grid_data[r][c] = None 
                texture_photo        = self.get_texture_for_label(None)  

                if self.cell_image_ids[r][c] is not None:
                    self.canvas.itemconfig(self.cell_image_ids[r][c], image=texture_photo)
                else:
                    x0, y0 = c * self.cell_size, r * self.cell_size
                    image_id = self.canvas.create_image(x0, y0, image=texture_photo, anchor="nw")
                    self.cell_image_ids[r][c] = image_id

                self.cell_images[r][c] = texture_photo
                self.canvas.tag_lower(self.cell_image_ids[r][c])

    def set_active_label(self, label):
        self.active_label = label
        self.label_var.set(label)
        print(f"Active label set to: {label}")

    def draw_grid(self):

        width  = self.cols * self.cell_size
        height = self.rows * self.cell_size
        
        for col in range(self.cols + 1):
            x = col * self.cell_size
            self.canvas.create_line(x, 0, x, height, fill="gray")

        for row in range(self.rows + 1):
            y = row * self.cell_size
            self.canvas.create_line(0, y, width, y, fill="gray")

    def on_click(self, event):
        self.paint_cell(event)

    def on_drag(self, event):
        self.paint_cell(event)

    def complete_grid(self):

        grid_numeric = np.zeros((self.rows, self.cols), dtype=np.int64)
        mask_numeric = np.zeros((self.rows, self.cols), dtype=np.int64)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid_data[r][c] is None:
                    grid_numeric[r, c] = self.num_labels  
                    mask_numeric[r, c] = 1
                else:

                    grid_numeric[r, c] = self.labels.index(self.grid_data[r][c])
        
        grid_tensor = torch.tensor(grid_numeric).unsqueeze(0)  
        mask_tensor = torch.tensor(mask_numeric).unsqueeze(0) 

        t_tensor   = torch.tensor([1000 - 1], dtype=torch.long)
        prompt     = self.text_input.get()
        print(f"Prompt: {prompt}")
        text_batch = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt")
            
        with torch.no_grad():
            output        = self.model(grid_tensor, t_tensor, text_batch, mask_tensor)
            output_tensor = output[0]  
            probs         = torch.softmax(output_tensor, dim=1)
            output_grid   = torch.argmax(probs, dim=1)
            output_grid   = output_grid[0].cpu().numpy() 
                        

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid_data[r][c] is None:
                    label_id = int(output_grid[r, c])
                    self.grid_data[r][c] = self.idx_to_label[label_id]

                    texture_photo = self.get_texture_for_label(self.grid_data[r][c])
                    x0 = c * self.cell_size
                    y0 = r * self.cell_size
                    
                    if self.cell_image_ids[r][c] is None:
                        image_id = self.canvas.create_image(x0, y0, image=texture_photo, anchor="nw")
                        self.cell_image_ids[r][c] = image_id
                    else:
                        image_id = self.cell_image_ids[r][c]
                        self.canvas.itemconfig(image_id, image=texture_photo)
                    self.cell_images[r][c] = texture_photo
                    self.canvas.tag_lower(image_id)

    def paint_cell(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        half_brush = self.brush_size // 2

        for dr in range(-half_brush, half_brush + 1):
            for dc in range(-half_brush, half_brush + 1):
                r = row + dr
                c = col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    if self.grid_data[r][c] != self.active_label:
                        self.grid_data[r][c] = self.active_label
                        texture_photo = self.get_texture_for_label(self.active_label)
                        x0 = c * self.cell_size
                        y0 = r * self.cell_size

                        if self.cell_image_ids[r][c] is None:
                            image_id = self.canvas.create_image(x0, y0, image=texture_photo, anchor="nw")
                            self.cell_image_ids[r][c] = image_id
                        else:
                            image_id = self.cell_image_ids[r][c]
                            self.canvas.itemconfig(image_id, image=texture_photo)

                        self.cell_images[r][c] = texture_photo
                        self.canvas.tag_lower(image_id)


    def get_texture_for_label(self, label):
        if label in self.textures:
            texture_image = self.textures[label]
            resized_texture = texture_image.resize((self.cell_size, self.cell_size), Image.NEAREST)
            return ImageTk.PhotoImage(resized_texture)
        else:
            blank = Image.new("RGBA", (self.cell_size, self.cell_size), "white")
            return ImageTk.PhotoImage(blank)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Cell Editor with Textures")
    editor = CellEditor(root, rows=64, cols=64, cell_size=8)
    root.mainloop()

from src.config import grid_cell_size, structure_labels, model, processor, device, bitmask_to_tile, textures
import pickle
import numpy as np
import torch
from PIL import Image


class Labeler:
    
    
    def __init__(self, image, text_labels):

        self.image              = image
        self.image_np           = np.array(image, dtype=np.uint8)
        self.grid_size          = (image.size[1] // grid_cell_size, image.size[0] // grid_cell_size)
        self.width              = self.grid_size[1] * grid_cell_size
        self.height             = self.grid_size[0] * grid_cell_size
        self.grid_labels        = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=object)
        self.width, self.height = image.size
        self.text_labels        = text_labels
        self.merged_groups      = {}

        with open('data/output/tile_mapping.pkl', 'rb') as f:
            self.tile_mapping = pickle.load(f)

    def get_grid_labels(self):
        return self.grid_labels
    
    def compute_final_regions(self, merged_groups):

        final_regions    = []
        image_h, image_w = self.image_np.shape[:2]  

        for group_labels in merged_groups.values():

            combined_mask        = (np.isin(self.assignment, group_labels)).astype(np.uint8) * 255    
            region_rgba          = np.zeros((image_h, image_w, 4), dtype=np.uint8)
            region_rgba[..., :3] = self.image
            region_rgba[..., 3]  = combined_mask

            final_regions.append((Image.fromarray(region_rgba)))
        return final_regions
    
    def find_best_tile(self, cell):

        query_embedding    = retrieve_image_embedding(cell)
        distances, indices = index.search(query_embedding, k=1)
        best_tile_idx      = indices[0][0]

        return self.tile_mapping[best_tile_idx], distances[0][0]
        
    def update_grid_labels(self, label_map):

        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                cell_labels            = label_map[y * grid_cell_size:(y + 1) * grid_cell_size, x * grid_cell_size:(x + 1) * grid_cell_size]
                unique, counts         = np.unique(cell_labels, return_counts=True)
                self.grid_labels[y, x] = unique[np.argmax(counts)]

    def output_labelling(self, flag):

        output_image = Image.new("RGB", (self.width, self.height))
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):

                label      = self.grid_labels[i, j]
                tile_image = textures[label].resize((grid_cell_size, grid_cell_size))
                output_image.paste(tile_image, (j * grid_cell_size, i * grid_cell_size))
        
        output_image.save("data/output/" + flag + "grid.jpg")
        with open("data/output/grid_labels.pkl", "wb") as f:
            pickle.dump(self.grid_labels, f)
        with open("data/output/label_to_masks.pkl", "wb") as f:
            pickle.dump(self.label_to_masks, f)
        if self.merged_groups:
            with open("data/output/merged_groups.pkl", "wb") as f:
                pickle.dump(self.merged_groups, f)
        self.image.save("data/output/map.jpg")


    def normal_merging(self, unique_labels, threshold):

        for label in unique_labels:
            for nbr in self.neighbors[label]:
                if label < nbr:  
                    root_label = self.find(label)
                    root_nbr = self.find(nbr)
                    if root_label != root_nbr:
                        sim = np.dot(self.group_embeddings[root_label],
                                    self.group_embeddings[root_nbr].T)[0, 0]
                        if sim > threshold:
                            union(label, nbr, self.parent)

    def improved_merging(self, unique_labels, threshold):

        changed = True
        while changed:
            changed = False
            best_match = {}
            for label in unique_labels:

                root_label = self.find(label)
                best_sim   = threshold  
                best_nbr   = None

                for nbr in self.neighbors[label]:
                    root_nbr = self.find(nbr)
                    if root_label == root_nbr:
                        continue

                    sim = np.dot(self.group_embeddings[root_label],
                                self.group_embeddings[root_nbr].T)[0, 0]
                    if sim > best_sim:

                        best_sim = sim
                        best_nbr = nbr

                if best_nbr is not None:
                    best_match[label] = (best_nbr, best_sim)

            for label, (nbr, sim) in best_match.items():
                if nbr in best_match and best_match[nbr][0] == label:
                    if self.find(label) != self.find(nbr):
                        self.union(label, nbr)
                        changed = True

    def calculate_neighbors(self, assignment, unique_labels):

        self.neighbors = {label: set() for label in unique_labels}
        h, w           = assignment.shape

        for i in range(h):
            for j in range(w):
                curr_label = assignment[i, j]
                if j + 1 < w:
                    nbr = assignment[i, j+1]
                    if nbr != curr_label:

                        self.neighbors[curr_label].add(nbr)
                        self.neighbors[nbr].add(curr_label)

                if i + 1 < h:
                    nbr = assignment[i+1, j]
                    if nbr != curr_label:

                        self.neighbors[curr_label].add(nbr)
                        self.neighbors[nbr].add(curr_label)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            self.parent[root_y]           = root_x
            self.group_segments[root_x].extend(self.group_segments[root_y])
            self.group_embeddings[root_x] = self.retrieve_image_embedding(self.image_from_labels(self.group_segments[root_x]))


    def image_from_mask(self, mask, image_np):

        y_idx, x_idx        = np.where(mask)
        cropped_region      = image_np[y_idx.min():y_idx.max()+1, x_idx.min():x_idx.max()+1]
        mask                = mask[y_idx.min():y_idx.max()+1, x_idx.min():x_idx.max()+1] * 255
        crop_rgba           = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

        crop_rgba[..., :3]  = cropped_region[..., :3]
        crop_rgba[..., 3]   = mask
        block_image         = Image.fromarray(crop_rgba)

        return block_image

    def image_from_labels(self, labels):
        
        y_inds, x_inds     = np.where(np.isin(self.assignment, labels))
        x_min, x_max       = x_inds.min(), x_inds.max()
        y_min, y_max       = y_inds.min(), y_inds.max()

        cropped_region     = self.image_np[y_min:y_max+1, x_min:x_max+1]
        mask               = np.isin(self.assignment[y_min:y_max+1, x_min:x_max+1], labels).astype(np.uint8) * 255 
        crop_rgba          = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

        crop_rgba[..., :3] = cropped_region[..., :3]
        crop_rgba[..., 3]  = mask
        block_image        = Image.fromarray(crop_rgba)

        return block_image


    @staticmethod
    def smooth_grid(grid_labels, text_labels):

        label_to_id    = {label: i for i, label in enumerate(text_labels)}
        id_to_label    = {i: label for label, i in label_to_id.items()}
        grid_label_ids = np.array([[label_to_id[label] for label in row] for row in grid_labels])
        grid_label_ids = np.array([
            [label_to_id.get(label, -1) for label in row]
            for row in grid_labels
        ])

        num_cells_x    = len(grid_labels[0])
        num_cells_y    = len(grid_labels)
        num_classes    = len(text_labels)
        unary          = np.zeros((len(grid_labels) * len(grid_labels[0]), num_classes), dtype=np.int32)

        for y in range(num_cells_y):
            for x in range(num_cells_x):
                label_idx                                 = grid_label_ids[y, x]
                if label_idx == -1:
                    unary[y * num_cells_x + x, :]         = 5 

                else:
                    unary[y * num_cells_x + x, :]         = 5  
                    unary[y * num_cells_x + x, label_idx] = 1


        pairwise = 5 * (1 - np.eye(num_classes, dtype=np.int32))

        edges = []
        for y in range(num_cells_y):
            for x in range(num_cells_x):
                idx = y * num_cells_x + x
                if x < num_cells_x - 1:
                    edges.append((idx, idx + 1))
                if y < num_cells_y - 1:
                    edges.append((idx, idx + num_cells_x))

        edges = np.array(edges, dtype=np.int32)

        smoothed_labels = pygco.cut_from_graph(
            edges, unary, pairwise, algorithm='swap'
        )

        return  np.array([[id_to_label[smoothed_labels[y * num_cells_x + x]]
                                for x in range(num_cells_x)]
                                for y in range(num_cells_y)])

    @staticmethod
    def crop_cell(image, i, j, block_size = 1):
        return image.crop((j * grid_cell_size, i * grid_cell_size, (j + block_size) * grid_cell_size, (i + block_size) * grid_cell_size))

    @staticmethod
    def auto_tile_swapping(tree_mask):

        rows = len(tree_mask)
        cols = len(tree_mask[0]) if rows > 0 else 0
        
        tile_map    = [[0]*cols for _ in range(rows)]
        neighbors_4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        for r in range(rows):
            for c in range(cols):
                if tree_mask[r][c] == 1:
                    bitmask = 0
                    for bit_index, (dr, dc) in enumerate(neighbors_4):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if tree_mask[nr][nc] == 1:
                                bitmask |= (1 << bit_index)
                    tile_index = bitmask_to_tile.get(bitmask, 0)
                    tile_map[r][c] = tile_index
                else:
                    tile_map[r][c] = 0
        
        return tile_map

    @staticmethod
    def retrieve_closest_label(query_features, text_labels, text_embeddings):

        similarities  = (query_features @ text_embeddings.T)[0]
        top_k_indices = similarities.argsort()[-1:][::-1]
        top_k_labels  = [text_labels[i] for i in top_k_indices]
        top_k_scores  = [similarities[i] for i in top_k_indices]
        
        return top_k_labels[0], top_k_scores[0]

    @staticmethod
    def compute_text_features(text_labels):
            
        inputs_text       = processor(text=text_labels, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            text_features = model.get_text_features(**inputs_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_embeddings   = text_features.cpu().numpy()

        return text_embeddings

    @staticmethod
    def retrieve_image_embedding(image):
        
        inputs             = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(inputs["pixel_values"])
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()

    def show_image(self, img, title="Image"):
        img.show(title=title)
        
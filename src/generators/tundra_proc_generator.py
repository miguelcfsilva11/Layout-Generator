import numpy as np
import random
import math
from collections import deque
from src.generators.proc_generator import ProcGenerator

class TundraProcGenerator(ProcGenerator):
    """
    Tundra‐themed generator, inheriting shared utilities from ProcGenerator.
    Features:
      - Base SNOW terrain
      - Small, frequent ICE patches scattered across snow
      - Very large MESA plateaus
      - Thin BRIDGE paths connecting mesas
      - WAYSTATION blobs placed atop each mesa
      - Occasional RESEARCH facility next to a mesa
    """

    def __init__(self, size=64, seed=None):
        super().__init__(size=size, seed=seed)
        self.title = "Tundra Kingdom"

    def generate_ice_patches(self):
        """
        Scatter many small ICE patches across the SNOW.
        Each patch is a make_blob of size 5–15, placed only on SNOW cells.
        We create 20–30 of these to simulate frequent icy spots.
        """
        patches = random.randint(10, 20)
        for _ in range(patches):

            attempts = 0
            while attempts < 50:
                cx = random.randint(0, self.size - 1)
                cy = random.randint(0, self.size - 1)
                if self.grid[cy, cx] == self.SNOW:
                    break
                attempts += 1
            else:
                continue  

            blob = self.make_blob(
                center           = (cx, cy),
                size             = random.randint(15, 30),
                elongation       = random.uniform(0.8, 1.2),
                rotation_degrees = random.uniform(0, 360)
            )

            valid = {pt for pt in blob if self.grid[pt[1], pt[0]] == self.SNOW}
            self.apply_terrain_to_grid(valid, self.ICE)

    def generate_mesas(self):
        """
        Generate 2–3 very large mesa plateaus (MESA). Returns a list of mesa blobs.
        Each mesa blob is a set of (x,y) cells. Mesas are size ~150–250.
        """
        mesa_blobs = []
        count = random.randint(2, 3)
        for _ in range(count): 
            cx   = random.randint(15, self.size - 16)
            cy   = random.randint(15, self.size - 16)
            blob = self.make_blob(
                center           = (cx, cy),
                size             = random.randint(150, 250),
                elongation       = random.uniform(0.8, 1.2),
                rotation_degrees = random.uniform(0, 20)
            )
            mesa_blobs.append(blob)
            self.apply_terrain_to_grid(blob, self.MESA)
        return mesa_blobs

    def generate_bridges_between_mesas(self, mesa_centroids):
        """
        For every pair of mesa centroids, carve a thin BRIDGE path between them.
        """
        for i in range(len(mesa_centroids)):
            for j in range(i + 1, len(mesa_centroids)):
                start = mesa_centroids[i]
                end   = mesa_centroids[j]
                path  = self.curved_path(start, end, curviness=0.1)
                for (px, py) in path:
                    if 0 <= px < self.size and 0 <= py < self.size:
                        if self.grid[py, px] in (self.SNOW, self.ICE):
                            self.grid[py, px] = self.BRIDGE

    def generate_waystations_on_mesas(self, mesa_blobs):
        """
        Place one WAYSTATION blob atop each mesa. WAYSTATION blob is size ~20–30,
        but only painted on cells that belong to the mesa.
        """
        for blob in mesa_blobs:
            centroid    = self.centroid(blob)
            wx, wy      = centroid
            ws_blob     = self.make_blob(
                center           = (wx, wy),
                size             = random.randint(20, 30),
                elongation       = random.uniform(0.8, 1.2),
                rotation_degrees = random.uniform(0, 360)
            )
            valid_ws = {pt for pt in ws_blob if pt in blob}
            self.apply_terrain_to_grid(valid_ws, self.WAYSTATION)

    def generate_research_facilities(self, mesa_blobs):
        """
        For each mesa, with 50% chance place one RESEARCH facility next to it.
        Facility blob is size ~20–30, painted on adjacent SNOW or ICE cells.
        """
        for blob in mesa_blobs:
            if random.random() > 0.5:
                continue
            perim = set()
            for (x, y) in blob:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if self.grid[ny, nx] in (self.SNOW, self.ICE):
                            perim.add((nx, ny))
            if not perim:
                continue
            cx, cy  = random.choice(list(perim))
            rf_blob = self.make_blob(
                center           = (cx, cy),
                size             = random.randint(20, 30),
                elongation       = random.uniform(0.8, 1.2),
                rotation_degrees = random.uniform(0, 360)
            )
            valid_rf = {pt for pt in rf_blob if self.grid[pt[1], pt[0]] in (self.SNOW, self.ICE)}
            self.apply_terrain_to_grid(valid_rf, self.RESEARCH)

    def generate_barren_tundra(self):
        """
        Barren tundra: fill with SNOW (already done), then optionally add:
          - Sparse ICE patches (~10% coverage)
          - A jagged mountain wall on a random edge
        """

        total_cells  = self.size * self.size
        min_count    = int(total_cells * 0.01)
        max_count    = int(total_cells * 0.15)
        target_count = random.randint(min_count, max_count)
        ice_cells    = set()

        while len(ice_cells) < target_count:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if self.grid[y, x] == self.SNOW:
                ice_cells.add((x, y))
        self.apply_terrain_to_grid(ice_cells, self.ICE)

        if random.random() < 0.3:
            self.set_on_edge(height=3, width=3, label=self.MOUNTAIN)

    def generate_map(self, seed):

        if seed is not None:
            self.seed = seed
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        self.grid[:] = self.SNOW

        if random.random() < 0.2:
            self.generate_barren_tundra()
            return

        self.generate_ice_patches()

        mesa_blobs     = self.generate_mesas()
        mesa_centroids = [self.centroid(blob) for blob in mesa_blobs]
        
        self.generate_bridges_between_mesas(mesa_centroids)
        self.generate_waystations_on_mesas(mesa_blobs)
        self.generate_research_facilities(mesa_blobs)

if __name__ == "__main__":
    generator = TundraProcGenerator(size=64, seed=None)
    generator.generate_map()
    generator.create_visualization()


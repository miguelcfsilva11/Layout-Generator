import numpy as np
import random
import math
from collections import deque
from src.generators.proc_generator import ProcGenerator

class DesertProcGenerator(ProcGenerator):
    """
    Desert‐themed generator, inheriting shared utilities from ProcGenerator.
    Modifications:
      - Mesa plateaus are larger.
      - If a road path would cross a mesa, the entire road is omitted.
      - Vegetation (palms) around oases is expanded (radius 2).
    """

    def __init__(self, size=64, seed=None):
        super().__init__(size=size, seed=seed)
        self.title = "Desert Kingdom"

    def generate_dune_fields(self, target_frac=0.25):
        """
        Generate dune ridges covering approximately `target_frac` of the map.
        Uses a biased BFS (favoring east-west expansion).
        Returns the set of all DUNE cells.
        """
        total_cells = self.size * self.size
        target_size = int(total_cells * target_frac)
        dune_set    = set()
        num_seeds   = random.randint(2, 4)
        seeds       = [(random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                 for _ in range(num_seeds)]
        frontier    = deque(seeds)
        dune_set.update(seeds)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while len(dune_set) < target_size and frontier:
            x, y = frontier.popleft()
            random.shuffle(directions)
            for dx, dy in directions:
                prob = 0.7 if abs(dx) == 1 else 0.3
                if random.random() > prob:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in dune_set:
                    dune_set.add((nx, ny))
                    frontier.append((nx, ny))
                    if len(dune_set) >= target_size:
                        break

        self.apply_terrain_to_grid(dune_set, self.DUNE)
        return dune_set

    def generate_rock_outcrops(self):
        """
        Generate 3–6 small rock blobs, paint as ROCK. Returns list of rock blobs.
        """
        rock_blobs = []
        for _ in range(random.randint(3, 6)):
            cx   = random.randint(5, self.size - 6)
            cy   = random.randint(5, self.size - 6)
            blob = self.make_blob(center=(cx, cy),
                                  size             = random.randint(30, 60),
                                  elongation       = random.uniform(0.8, 1.2),
                                  rotation_degrees = random.uniform(0, 360))
            rock_blobs.append(blob)
            self.apply_terrain_to_grid(blob, self.ROCK)
        return rock_blobs

    def generate_mesas(self, rock_blobs):
        """
        Generate 1–3 large mesa plateaus (MESA), typically built around a rock outcrop or random center.
        Now using larger size ranges to produce bigger mesas.
        Returns a list of mesa blobs.
        """
        mesa_blobs = []
        for _ in range(random.randint(1, 3)):
            if rock_blobs and random.random() < 0.7:
                base_blob = random.choice(rock_blobs)
                cx, cy    = self.centroid(base_blob)
            else:   
                cx        = random.randint(15, self.size - 16)
                cy        = random.randint(15, self.size - 16)

            blob = self.make_blob(center=(cx, cy),
                                  size=random.randint(120, 200),
                                  elongation=random.uniform(0.8, 1.2),
                                  rotation_degrees=random.uniform(0, 20))
            mesa_blobs.append(blob)
            self.apply_terrain_to_grid(blob, self.MESA)
        return mesa_blobs

    def generate_oases(self):
        """
        Generate 1–3 oasis blobs (WATER) using make_blob for organic shapes.
        Returns a list of (ox, oy, blob_cells) tuples.
        """
        oasis_list = []
        for _ in range(random.randint(1, 3)):
            ox = random.randint(5, self.size - 6)
            oy = random.randint(5, self.size - 6)
            oasis_blob = self.make_blob(center=(ox, oy),
                                       size=random.randint(30, 60),
                                       elongation=random.uniform(0.7, 1.3),
                                       rotation_degrees=random.uniform(0, 360))
            self.apply_terrain_to_grid(oasis_blob, self.WATER)
            oasis_list.append((ox, oy, oasis_blob))
        return oasis_list

    def generate_palms_around_oases(self, oasis_list):
        """
        For each oasis_blob, create a wider ring of FORE forest cells around it
        to simulate larger palm groves (radius = 2).
        """
        for (_, _, oasis_blob) in oasis_list:
            expanded  = self.expand_area_around_points(oasis_blob, radius=2)
            potential = {pt for pt in expanded if self.grid[pt[1], pt[0]] == self.SAND}
            for (x, y) in potential:
                if random.random() < 0.6:
                    self.grid[y, x] = self.FOREST

    def generate_desert_keep(self, oasis_list, mesa_blobs):
        """
        Place a single desert keep (CASTLE) with 80% probability.
        Prefer a random MESA if available; otherwise place on/near a random oasis.
        Returns (keep_blob, keep_centroid) or (None, None).
        """
        if random.random() > 0.8:
            return None, None

        while True:
            if mesa_blobs and random.random() < 0.7:
                base_blob = random.choice(mesa_blobs)
                cx, cy    = self.centroid(base_blob)
            elif oasis_list:
                ox, oy, _ = random.choice(oasis_list)
                cx, cy    = (ox, oy)
            else:
                cx        = random.randint(5, self.size - 6)
                cy        = random.randint(5, self.size - 6)

            castle_blob = self.make_blob(center=(cx, cy),
                                        size=random.randint(40, 80),
                                        elongation=random.uniform(1.0, 2.0),
                                        rotation_degrees=random.uniform(0, 360))

            if not any(self.grid[y, x] == self.WATER for (x, y) in castle_blob):
                self.apply_terrain_to_grid(castle_blob, self.CASTLE)
                return castle_blob, self.centroid(castle_blob)

    def generate_desert_villages(self, oasis_list, castle_centroid):
        """
        Generate 1–3 desert towns (VILLAGE) with 70% probability.
        Each must be within ~15 cells of an oasis, ≥15 cells from the keep if it exists,
        and not overlapping WATER. Returns list of centroids.
        """
        town_centroids = []
        if random.random() > 0.7:
            return town_centroids

        attempts   = 0
        target_num = random.randint(1, 3)

        while len(town_centroids) < target_num and attempts < 200:
            vx = random.randint(5, self.size - 6)
            vy = random.randint(5, self.size - 6)
            town_blob = self.make_blob(center=(vx, vy),
                                      size=random.randint(80, 120),
                                      elongation=random.uniform(0.7, 1.5),
                                      rotation_degrees=random.uniform(0, 360))
            tc             = self.centroid(town_blob)

            close_to_oasis = any(self.distance(tc, (ox, oy)) <= 15
                                 for (ox, oy, _) in oasis_list)
            overlaps_water = any(self.grid[y, x] == self.WATER for (x, y) in town_blob)
            if castle_centroid:
                dist_to_keep = self.distance(tc, castle_centroid) < 15
            else:
                dist_to_keep = False
            too_close_to_other = any(self.distance(tc, other) < 15 for other in town_centroids)

            if close_to_oasis and not overlaps_water and not dist_to_keep and not too_close_to_other:
                self.apply_terrain_to_grid(town_blob, self.VILLAGE)
                town_centroids.append(tc)

            attempts += 1

        return town_centroids

    def generate_cliffs(self, mesa_blobs):
        """
        Any cell adjacent 4-directionally to a MESA cell (but not itself MESA)
        becomes a CLIFF. Paint those as CLIFF.
        """
        for mesa in mesa_blobs:
            for (x, y) in mesa:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if self.grid[ny, nx] not in (self.MESA, self.CLIFF):
                            self.grid[ny, nx] = self.CLIFF

    def generate_caravan_routes(self, castle_centroid, town_centroids):
        """
        Draw thin (single‐cell) caravan routes between keep and villages, if both exist.
        If any cell of a path lands on MESA, skip that entire path. Otherwise:
          - WATER → BRIDGE
          - DUNE or SAND → ROAD
        """
        if not castle_centroid or not town_centroids:
            return

        POIs = [castle_centroid] + town_centroids
        for i in range(len(POIs)):
            for j in range(i + 1, len(POIs)):
                start = POIs[i]
                end   = POIs[j]
                path  = self.curved_path(start, end, curviness=0.2)

                if any(0 <= px < self.size and 0 <= py < self.size and self.grid[py, px] == self.MESA
                       for (px, py) in path):
                    continue

                for (px, py) in path:
                    if not (0 <= px < self.size and 0 <= py < self.size):
                        continue
                    cell = self.grid[py, px]
                    if cell in (self.VILLAGE, self.CASTLE):
                        continue
                    if cell == self.WATER:
                        self.grid[py, px] = self.BRIDGE
                    else:
                        self.grid[py, px] = self.ROAD

    def generate_waystations(self, castle_centroid, town_centroids):
        """
        If no villages or keep exist, drop 1-2 larger WAYSTATION blobs randomly.
        Otherwise, for any caravan route > 20 cells, place a WAYSTATION blob at midpoint.
        WAYSTATION blobs are generated via make_blob with size ~25-40,
        but only paint on SAND (do not overwrite MESA).
        """
        waystations = []

        if not castle_centroid and not town_centroids:
            num_ws = random.randint(1, 2)
            for _ in range(num_ws):
                wx      = random.randint(10, self.size - 11)
                wy      = random.randint(10, self.size - 11)
                ws_blob = self.make_blob(center=(wx, wy),
                                         size             = random.randint(25, 40),
                                         elongation       = random.uniform(0.8, 1.2),
                                         rotation_degrees = random.uniform(0, 360))
                valid_ws = {pt for pt in ws_blob
                            if self.grid[pt[1], pt[0]] == self.SAND}
                self.apply_terrain_to_grid(valid_ws, self.WAYSTATION)
                waystations.append((wx, wy))
        else:
            if castle_centroid and town_centroids:
                for tc in town_centroids:
                    dist = int(self.distance(castle_centroid, tc))
                    if dist > 20:
                        midx    = (castle_centroid[0] + tc[0])         // 2
                        midy    = (castle_centroid[1] + tc[1])         // 2
                        ws_blob = self.make_blob(center=(midx, midy),
                                                 size             = random.randint(25, 40),
                                                 elongation       = random.uniform(0.8, 1.2),
                                                 rotation_degrees = random.uniform(0, 360))
                        valid_ws = {pt for pt in ws_blob
                                    if self.grid[pt[1], pt[0]] == self.SAND}
                        self.apply_terrain_to_grid(valid_ws, self.WAYSTATION)
                        waystations.append((midx, midy))

        return waystations

    def generate_barren_desert(self):
        """
        The grid is already filled with SAND at this point.
        Optionally add:
          - A light dune field (~15% coverage)
          - A jagged mountain wall on a random edge
        """
        if random.random() < 0.6:
            self.generate_dune_fields(target_frac=0.15)

        if random.random() < 0.2:
            self.set_on_edge(height=3, width=3, label=self.MOUNTAIN)



    def generate_map(self, seed=None):

        self.grid[:] = self.SAND
        if seed is not None:
            self.seed = seed
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        choice = random.random()
        if choice < 0.2:
            self.grid[:] = self.SAND
            self.generate_barren_desert()
            return
        else:
            dune_set   = self.generate_dune_fields()
            rock_blobs = self.generate_rock_outcrops()
            mesa_blobs = self.generate_mesas(rock_blobs)
            oasis_list = self.generate_oases()

            self.generate_palms_around_oases(oasis_list)

            castle_blob, castle_centroid = self.generate_desert_keep(oasis_list, mesa_blobs)
            town_centroids               = self.generate_desert_villages(oasis_list, castle_centroid)

            if random.random() < 0.5: 
                rx        = random.randint(10, self.size - 11)
                ry        = random.randint(10, self.size - 11)
                ruin_blob = self.make_blob(center=(rx, ry),
                                        size             = random.randint(20, 30),
                                        elongation       = random.uniform(0.9, 1.1),
                                        rotation_degrees = random.uniform(0, 360))
                rc                = self.centroid(ruin_blob)
                too_close_to_keep = castle_centroid and self.distance(rc, castle_centroid) < 20
                too_close_to_town = any(self.distance(rc, tc) < 20 for tc in town_centroids)
                if not too_close_to_keep and not too_close_to_town: 
                    self.apply_terrain_to_grid(ruin_blob, self.VILLAGE)
                    town_centroids.append(rc)

            self.generate_cliffs(mesa_blobs)
            self.generate_caravan_routes(castle_centroid, town_centroids)
            self.generate_waystations(castle_centroid, town_centroids)



if __name__ == "__main__":
    generator = DesertProcGenerator(size=64, seed=None)
    generator.generate_map()
    generator.create_visualization()

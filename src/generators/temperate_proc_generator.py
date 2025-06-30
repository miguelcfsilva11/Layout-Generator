import numpy as np
import random
import math
from collections import deque
from src.generators.proc_generator import ProcGenerator

class TemperateProcGenerator(ProcGenerator):

    def __init__(self, size=64, seed=None):
        super().__init__(size=size, seed=seed)
        self.title = "Woodland Kingdom"
    def generate_road_only(self):


        if random.random() < 0.5:
            edge1 = (0, random.randint(0, self.size - 1))
            edge2 = (self.size - 1, random.randint(0, self.size - 1))
        else:
            edge1 = (random.randint(0, self.size - 1), 0)
            edge2 = (random.randint(0, self.size - 1), self.size - 1)
        road_path = self.curved_path(edge1, edge2, curviness=0.1)

        for (rx, ry) in road_path:
            road_area = self.expand_area_around_points([(rx, ry)], radius=1)
            for (nx, ny) in road_area:
                cell = self.grid[ny, nx]
                if cell in (self.GRASS, self.FOREST):
                    self.grid[ny, nx] = self.ROAD
                elif cell == self.WATER:
                    self.grid[ny, nx] = self.BRIDGE

        if random.random() < 0.5:
            mid        = road_path[len(road_path) // 2]
            wx, wy     = mid
            ws_blob    = self.make_blob(center=(wx, wy), size=random.randint(9, 16),
                                    elongation       = random.uniform(0.8, 1.2),
                                    rotation_degrees = random.uniform(0, 360))
            valid      = {pt for pt in ws_blob if self.grid[pt[1], pt[0]] == self.GRASS}

    def generate_forest_glade(self):

        total_cells = self.size * self.size
        target_size = random.randint(int(total_cells * 0.4), int(total_cells * 0.6))
        forest      = set()
        num_seeds   = random.randint(3, 5)
        seeds       = [(random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                for _ in range(num_seeds)]
        forest.update(seeds)
        frontier    = deque(seeds)
        directions  = [(-1, 0), (1, 0), (0, -1), (0, 1)]


        while len(forest) < target_size and frontier:
            x, y = frontier.popleft()
            random.shuffle(directions)
            for dx, dy in directions:
                if random.random() > 0.8:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in forest:
                    forest.add((nx, ny))
                    frontier.append((nx, ny))
                    if len(forest) >= target_size:
                        break
        
        self.apply_terrain_to_grid(forest, self.FOREST)

    def generate_lakeshore(self):

        lake_center = (
            random.randint(self.size // 3, 2 * self.size // 3),
            random.randint(self.size // 3, 2 * self.size // 3)
        )

        lake_blob   = set()
        radius_main = random.randint(self.size // 3, self.size // 3)
        radius_tail = radius_main                                     // 1.2

        cx, cy = lake_center

        for y in range(self.size):
            for x in range(self.size):

                dx        = x - cx
                dy        = y - cy
                dist_main = (dx ** 2) / (radius_main ** 2) + (dy ** 2) / (radius_main ** 2)
                dist_tail = (dx ** 2) / (radius_tail ** 2) + ((dy + radius_main) ** 2) / (radius_tail ** 2)

                if dist_main <= 1 or dist_tail <= 1:
                    lake_blob.add((x, y))

        
        rotated_blob   = self.rotate_blob(lake_blob, random.uniform(0, 360), self.size)
        xs             = [pt[0] for pt in rotated_blob]
        ys             = [pt[1] for pt in rotated_blob]
        new_cx         = int(sum(xs) / len(xs))
        new_cy         = int(sum(ys) / len(ys))
        rotated_center = (new_cx, new_cy)

        self.apply_terrain_to_grid(rotated_blob, self.WATER)

        edge_points = [
            [(x, 0) for x in range(self.size)],               
            [(self.size - 1, y) for y in range(self.size)],   
            [(x, self.size - 1) for x in range(self.size)],   
            [(0, y) for y in range(self.size)],              
        ]

        river_start = random.choice(random.choice(edge_points))
        river_path  = self.curved_path(river_start, rotated_center, curviness=0.2)
        
        for (x, y) in river_path:
            river_area = self.expand_area_around_points([(x, y)], radius=2)
            for (rx, ry) in river_area:
                if 0 <= rx < self.size and 0 <= ry < self.size:
                    self.grid[ry, rx] = self.WATER


    def generate_mountain_pass(self):
        self.set_on_edge(height=5, width=5, label=self.MOUNTAIN)

        mountain_cells = [(x, y) for y in range(self.size) for x in range(5) if self.grid[y, x] == self.MOUNTAIN]
        if not mountain_cells:
            return

        start = random.choice(mountain_cells)
        candidates = [(x, y) for y in range(self.size) for x in range(self.size // 2, self.size) if self.grid[y, x] in (self.GRASS, self.FOREST)]
        if not candidates:
            return
        end = random.choice(candidates)




    def generate_forest_terrain(self, target_frac=0.2):

        total_cells = self.size * self.size
        target_size = random.randint(int(total_cells * target_frac),
                                     int(total_cells * 0.6))
        forest      = set()
        num_seeds   = random.randint(2, 4)
        seeds       = [(random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                    for _ in range(num_seeds)]
        frontier    = deque(seeds)
        forest.update(seeds)
        directions  = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while len(forest) < target_size and frontier:
            x, y = frontier.popleft()
            random.shuffle(directions)
            for dx, dy in directions:
                if random.random() > 0.7:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in forest:
                    forest.add((nx, ny))
                    frontier.append((nx, ny))
                    if len(forest) >= target_size:
                        break

        self.apply_terrain_to_grid(forest, self.FOREST)
        return forest

    def generate_winding_river(self, min_distance=0.6):
        """
        Pick two points on opposite edges with distance ≥ min_distance*self.size.
        Carve a curved path between them and expand to width=2. Paint as WATER.
        Returns the set of all WATER cells.
        """
        edges = [
            [(x, 0) for x in range(self.size)],
            [(self.size - 1, y) for y in range(self.size)],
            [(x, self.size - 1) for x in range(self.size)],
            [(0, y) for y in range(self.size)],
        ]

        while True:
            e1, e2   = random.sample(range(4), 2)
            start    = random.choice(edges[e1])
            end      = random.choice(edges[e2])
            if self.distance(start, end) >= min_distance * self.size:
                break

        path        = self.curved_path(start, end, curviness=0.1)
        river_width = 2
        river_area  = self.expand_area_around_points(path, river_width)
        self.apply_terrain_to_grid(river_area, self.WATER)
        return set(river_area)

    def generate_castle_terrain(self, river_set):

        while True:
            cx          = random.randint(5, self.size - 6)
            cy          = random.randint(5, self.size - 6)
            castle_blob = self.make_blob(center=(cx, cy),
                                        size             = random.randint(50, 100),
                                        elongation       = random.uniform(0.5, 2.0),
                                        rotation_degrees = random.uniform(0, 360))
            if not self.overlaps_terrain(castle_blob, river_set):
                break
        self.apply_terrain_to_grid(castle_blob, self.CASTLE)
        return castle_blob, self.centroid(castle_blob)

    def generate_village_terrain(self, river_set, castle_centroid):

        village_centroids = []
        num_villages      = random.randint(1, 2)
        min_dist          = 0.3 * self.size
        attempts          = 0

        while len(village_centroids) < num_villages and attempts < 200:
            vx     = random.randint(5, self.size - 6)
            vy     = random.randint(5, self.size - 6)
            v_blob = self.make_blob(center=(vx, vy),
                                    size             = random.randint(150, 200),
                                    elongation       = random.uniform(0.5, 2.0),
                                    rotation_degrees = random.uniform(0, 360))
            vc        = self.centroid(v_blob)
            too_close = (self.distance(vc, castle_centroid) < min_dist or
                         any(self.distance(vc, other) < min_dist for other in village_centroids))

            if not too_close and not self.overlaps_terrain(v_blob, river_set): 
                self.apply_terrain_to_grid(v_blob, self.VILLAGE)
                village_centroids.append(vc)

            attempts += 1

        return village_centroids

    def generate_road_terrain(self, castle_centroid, village_centroids, river_set):

        def apply_path(path):
            for (i, j) in path:
                road_area = self.expand_area_around_points([(i, j)], radius=1)
                for (ni, nj) in road_area:
                    cell = self.grid[nj, ni]
                    if cell in (self.VILLAGE, self.CASTLE):
                        continue
                    if (ni, nj) in river_set:
                        self.grid[nj, ni] = self.BRIDGE
                    else:
                        self.grid[nj, ni] = self.ROAD

        for vc in village_centroids:
            path = self.curved_path(castle_centroid, vc, curviness=0.1)
            apply_path(path)

        if not np.any(self.grid == self.BRIDGE) and random.random() < 0.5:
            edges = [
                [(x, 0) for x in range(self.size)],
                [(self.size - 1, y) for y in range(self.size)],
                [(x, self.size - 1) for x in range(self.size)],
                [(0, y) for y in range(self.size)],
            ]
            for _ in range(10):
                e_idx      = random.randrange(4)
                edge_pt    = random.choice(edges[e_idx])
                extra_path = self.curved_path(castle_centroid, edge_pt, curviness=0.1)

                def extend_to_edge(pt, idx, amt=3):
                    x, y = pt
                    if idx == 0:
                        return [(x, max(0, y - i)) for i in range(1, amt + 1)]
                    if idx == 1:
                        return [(min(self.size - 1, x + i), y) for i in range(1, amt + 1)]
                    if idx == 2:
                        return [(x, min(self.size - 1, y + i)) for i in range(1, amt + 1)]
                    if idx == 3:
                        return [(max(0, x - i), y) for i in range(1, amt + 1)]
                    return []

                extra_path.extend(extend_to_edge(edge_pt, e_idx))
                if any(pt in river_set for pt in extra_path):
                    apply_path(extra_path)
                    break

    def generate_windmill_terrain(self, river_set):

        def try_place(rx, ry):
            setback = random.choice([1, 2])
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: 

                cx      = rx + dx * (setback + 1)
                cy      = ry + dy * (setback + 1)
                start_x = cx - 2
                start_y = cy - 2

                if (0 <= start_x < self.size and 0 <= start_y < self.size
                        and start_x + 3 < self.size and start_y + 3 < self.size):
                    area = [(start_x + i, start_y + j) for i in range(4) for j in range(4)]
                    if all(self.grid[y, x] in (self.GRASS, self.FOREST) for (x, y) in area): 
                        self.apply_terrain_to_grid(area, self.WAYSTATION)
                        return True
            return False

        num_windmills = random.randint(1, 3)
        placed        = 0
        river_list    = list(river_set)
        random.shuffle(river_list)

        for (rx, ry) in river_list:
            if placed >= num_windmills:
                break
            if try_place(rx, ry):
                placed += 1

    def generate_map(self, seed):

        if seed is not None: 
            self.seed = seed
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        self.grid[:] = self.GRASS

        choice = random.random()
        if choice < 0.65: 

            forest                        = self.generate_forest_terrain()
            river                         = self.generate_winding_river()
            castle_blob, castle_centroid  = self.generate_castle_terrain(river)
            village_centroids             = self.generate_village_terrain(river, castle_centroid)

            self.generate_road_terrain(castle_centroid, village_centroids, river)
            self.generate_windmill_terrain(river)

        elif choice < 0.75:
            if random.random() < 0.5:
                forest = self.generate_forest_terrain(target_frac=0.4)
            if random.random() < 0.5:
                river = self.generate_winding_river()
            if random.random() < 0.5:
                self.generate_road_only()

        elif choice < 0.85:
            self.generate_forest_glade()
            if random.random() < 0.5:
                self.generate_lakeshore()
        else:
            self.generate_forest_terrain(target_frac=0.2)
            self.generate_mountain_pass()


if __name__ == "__main__":

    generator = TemperateProcGenerator(size=64, seed=42)
    generator.generate_map()
    generator.create_visualization()
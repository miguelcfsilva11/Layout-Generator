import numpy as np
import random
import math
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.widgets import Button
from scipy.ndimage import rotate, binary_fill_holes


class ProcGenerator:
    """
    Abstract base class for any procedural map generator.
    Contains shared utility methods: blob creation, path carving, expansions, etc.
    Subclasses must override `generate_map(...)` to implement biome-specific logic,
    and may optionally override `create_visualization(...)` if they need a different palette.
    """

    SAND        = 0   # Empty desert sand
    WATER       = 1   # Oasis water / snow melt
    GRASS       = 2   # Grass patches
    SNOW        = 3   # Base snowy terrain
    ICE         = 4   # Small ice patches
    FOREST      = 5   # Palm groves, shrubs, ghost wood
    VILLAGE     = 6   # Desert town / ruin / temple
    CASTLE      = 7   # Desert keep (fortress)
    WAYSTATION  = 8   # Caravan rest stop (desert or snowy)
    ROAD        = 9   # Caravan route (thin)
    BRIDGE      = 10  # Causeway over water or mesa edge
    RESEARCH    = 11  # Research facility / outpost next to mesa
    CLIFF       = 12  # Cliff edge around a mesa
    DUNE        = 13  # Sand ridges
    ROCK        = 14  # Rocky outcrops
    MESA        = 15  # Mesa plateaus
    MOUNTAIN    = 16  # Large rocky terrain (mountains)

    labels = [
        SAND, WATER, GRASS, SNOW, ICE,
        FOREST, VILLAGE, CASTLE, WAYSTATION,
        ROAD, BRIDGE, RESEARCH, CLIFF,
        DUNE, ROCK, MESA, MOUNTAIN
    ]
    
    def __init__(self, size=64, seed=None):
        """
        Initialize the generator with a grid size and (optional) random seed.
        """
        self.size = size
        if seed is None:
            seed  = random.randint(0, 1_000_000)
        self.seed = seed

        random.seed(self.seed)
        np.random.seed(self.seed)
        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)

    @staticmethod
    def rotate_blob(blob, angle_degrees, grid_size):
        """
        Rotate the blob mask around its center by a given angle, filling gaps.
        """
        if angle_degrees == 0:
            return blob

        grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        for x, y in blob:
            grid[y, x] = 1

        pad            = grid_size                                                        // 2
        padded         = np.pad(grid, pad_width=pad, mode='constant', constant_values=0)
        rotated        = rotate(padded, angle=angle_degrees, reshape=False, order=1)
        rotated        = (rotated > 0.2).astype(np.uint8)
        rotated_filled = binary_fill_holes(rotated).astype(np.uint8)
        center         = rotated_filled.shape[0]                                          // 2
        half           = grid_size                                                        // 2
        cropped        = rotated_filled[
            center - half: center + half,
            center - half: center + half
        ]

        new_blob = set()
        ys, xs = np.nonzero(cropped)
        for y, x in zip(ys, xs):
            new_blob.add((x, y))
        return new_blob

    def make_blob(self, center, size, elongation=1.0, rotation_degrees=0.0):
        """
        Create a roughly rectangular blob (width x height ≈ sqrt(size) x sqrt(size)),
        optionally elongated, then rotated by rotation_degrees.
        Returns a set of (x, y) coordinates. Does not modify self.grid.
        """
        cx, cy = center
        base   = math.sqrt(size)

        if elongation >= 1.0:
            width  = int(base * elongation)
            height = int(base)
        else:
            width  = int(base)
            height = int(base / elongation)

        width  = max(1, min(width, self.size))
        height = max(1, min(height, self.size))
        left   = max(0, cx - width // 2)
        right  = min(self.size, left + width)
        top    = max(0, cy - height // 2)
        bottom = min(self.size, top + height)

        blob = set()
        for x in range(left, right):
            for y in range(top, bottom):
                blob.add((x, y))

        return self.rotate_blob(blob, rotation_degrees, self.size)

    @staticmethod
    def distance(a, b):
        """Euclidean distance between two (x,y) points."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def centroid(blob):
        """Compute the integer centroid of a set of (x,y) points."""
        if not blob:
            raise ValueError("centroid() called with empty blob")
        xs = [p[0] for p in blob]
        ys = [p[1] for p in blob]
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    @staticmethod
    def bresenham(a, b):
        """Bresenham’s line algorithm between two points a=(x0,y0), b=(x1,y1)."""
        (x0, y0), (x1, y1) = a, b

        pts = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            pts.append((x0, y0))
            if (x0, y0) == (x1, y1):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return pts

    def curved_path(self, start, end, curviness=0.3):
        """
        Generate a curved path from start=(x0,y0) to end=(x1,y1) by
        inserting a few random “perpendicular” offsets at intermediate waypoints,
        then running Bresenhams line between them.
        """
        (x0, y0), (x1, y1) = start, end
        dx                 = x1 - x0
        dy                 = y1 - y0
        dist               = math.hypot(dx, dy)
        num_waypoints      = max(2, int(dist * 0.1))
        waypoints          = [start]

        for i in range(1, num_waypoints): 
            t     = i / num_waypoints
            mid_x = x0 + t * dx
            mid_y = y0 + t * dy
            if dist > 0: 
                perp_x = -dy / dist
                perp_y = dx / dist
            else: 
                perp_x, perp_y = 0, 0

            offset = random.uniform(-curviness * dist, curviness * dist)
            cx     = int(mid_x + offset * perp_x)
            cy     = int(mid_y + offset * perp_y)
            cx     = max(0, min(cx, self.size - 1))
            cy     = max(0, min(cy, self.size - 1))

            waypoints.append((cx, cy))

        waypoints.append(end)
        full_path = []
        for i in range(len(waypoints) - 1):
            segment = self.bresenham(waypoints[i], waypoints[i + 1])
            if i > 0:
                segment = segment[1:]
            full_path.extend(segment)

        return full_path

    def expand_area_around_points(self, points, radius):
        """
        Return a set of all grid points within a Chebyshev-distance radius of any point in `points`.
        (This yields square expansions; if you wanted diamond/circle, modify accordingly.)
        """
        expanded = set()
        for (px, py) in points:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        expanded.add((nx, ny))
        return expanded

    def apply_terrain_to_grid(self, positions, terrain_type):
        """
        Paint each (x,y) in `positions` onto self.grid with `terrain_type`.
        """
        for (x, y) in positions:
            if 0 <= x < self.size and 0 <= y < self.size:
                self.grid[y, x] = terrain_type

    def overlaps_terrain(self, blob, terrain_set):
        """
        Return True if any (x,y) in `blob` is in `terrain_set`.
        """
        return any((x, y) in terrain_set for (x, y) in blob)

    def generate_map(self):
        """
        Abstract method: generate self.grid (size x size) with the biome-specific logic.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_map()")

    def create_visualization(self, title="Procedural Map"):
        """
        Visualize self.grid using Matplotlib. Requires:
          - color_dict: { label_int: hex_color_string, ... }
          - legend_names: { label_int: display_name, ... }
        """

        color_dict = {
            self.SAND:        '#EEDC82',  # Sand
            self.WATER:       '#4A8FD4',  # Water / Oasis
            self.GRASS:       '#A8D08D',  # Grass
            self.SNOW:        '#FFFFFF',  # Snow
            self.ICE:         '#B0E0E6',  # Ice Patches

            self.FOREST:      '#2E7D32',  # Forest / Vegetation

            self.VILLAGE:     '#D32F2F',  # Village / Temple
            self.CASTLE:      '#A1887F',  # Castle / Keep
            self.WAYSTATION:  '#FFB74D',  # Waystation / Rest Stop

            self.ROAD:        '#8D6E63',  # Road
            self.BRIDGE:      '#FFD700',  # Bridge / Causeway
            self.RESEARCH:    '#7B68EE',  # Research Outpost

            self.CLIFF:       '#A0522D',  # Cliff
            self.DUNE:        '#D2B48C',  # Dune
            self.ROCK:        '#8B8680',  # Rock
            self.MESA:        '#C19A6B',  # Mesa
            self.MOUNTAIN:    '#6D4C41',  # Mountain
        }

        legend_names = {
            self.SAND:        'Sand',
            self.WATER:       'Water / Oasis',
            self.GRASS:       'Grass',
            self.SNOW:        'Snow',
            self.ICE:         'Ice Patch',

            self.FOREST:      'Forest / Vegetation',

            self.VILLAGE:     'Village / Ruin / Temple',
            self.CASTLE:      'Castle / Keep',
            self.WAYSTATION:  'Waystation',

            self.ROAD:        'Caravan Route',
            self.BRIDGE:      'Bridge / Causeway',
            self.RESEARCH:    'Research Outpost',

            self.CLIFF:       'Cliff Edge',
            self.DUNE:        'Dune Ridge',
            self.ROCK:        'Rocky Outcrop',
            self.MESA:        'Mesa Plateau',
            self.MOUNTAIN:    'Mountain Range',
        }


        max_label  = max(color_dict.keys())
        color_list = [color_dict[i] for i in range(max_label + 1)]
        cmap       = ListedColormap(color_list)
        norm       = BoundaryNorm(np.arange(max_label + 2) - 0.5, cmap.N)
        fig, ax    = plt.subplots(figsize=(8, 8))
        im         = ax.imshow(self.grid, cmap=cmap, norm=norm)

        ax.set_title(title, fontsize=14)
        ax.axis('off')

        legend_elems = [Patch(facecolor=color_dict[label], label=legend_names[label])
                        for label in sorted(color_dict.keys())]
        ax.legend(handles=legend_elems, loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

        title_text = ax.set_title(f"{title} — Seed: {self.seed}", fontsize=14)
        button_ax  = plt.axes([0.82, 0.02, 0.15, 0.04])
        button     = Button(button_ax, 'Regenerate Map')

        def regenerate(event):

            self.regenerate_map()
            im.set_array(self.grid)
            title_text.set_text(f"{title} — Seed: {self.seed}")

            plt.draw()

        button.on_clicked(regenerate)
        plt.tight_layout()
        plt.show()

    def regenerate_map(self):
        """
        Abstract hook: used by the 'Regenerate' button. By default, just calls generate_map() with a new random seed.
        Subclasses may override if they need to reuse class state differently.
        """
        self.seed = random.randint(0, 1_000_000)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.generate_map()
    def set_on_edge(self, height=1, width=1, label=0):
        """
        Create a jagged edge between two random perimeter points (possibly diagonal),
        then fill the correct side as `label`.
        Returns (start, end) that were used.
        """
        top_left      = (0, 0)
        top_middle    = (self.size // 2, 0)
        top_right     = (self.size - 1, 0)
        right_middle  = (self.size - 1, self.size // 2)
        bottom_right  = (self.size - 1, self.size - 1)
        bottom_middle = (self.size // 2, self.size - 1)
        bottom_left   = (0, self.size - 1)
        left_middle   = (0, self.size // 2)

        points = [
            top_left, top_middle, top_right,
            right_middle,
            bottom_right, bottom_middle, bottom_left,
            left_middle
        ]

        start, end = random.sample(points, 2)
        self.create_mountain_edge(start, end, height, width, label)
        return (start, end)

    def create_mountain_edge(self, start, end, height, width, label):
        """
        Build a jagged “wall” between `start` and `end` (using curved_path),
        thicken it by (width, height), then flood‐fill from one side to label
        the “inside” as `label`. This version picks a seed by stepping
        perpendicular to the first wall segment.
        """
        path = self.curved_path(start, end, curviness=0.1)
        edge_pixels = set()
        for (x, y) in path:
            for dx in range(-width // 2, width // 2 + 1):
                for dy in range(-height // 2, height // 2 + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        edge_pixels.add((nx, ny))


        if len(path) < 2:
            return 

        (x0, y0), (x1, y1) = path[0], path[1]
        dx, dy             = x1 - x0, y1 - y0
        perp1              = (-dy, dx)
        perp2              = (dy, -dx)
        floodfill_seed     = None

        for px, py in (perp1, perp2):
            sx, sy = x0 + px, y0 + py
            while 0 <= sx < self.size and 0 <= sy < self.size and (sx, sy) in edge_pixels:
                sx += px
                sy += py
            if 0 <= sx < self.size and 0 <= sy < self.size and (sx, sy) not in edge_pixels:
                floodfill_seed = (sx, sy)
                break

        if floodfill_seed is None:
            for corner in [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]:
                if corner not in edge_pixels:
                    floodfill_seed = corner
                    break
            if floodfill_seed is None:
                return 

        visited = set(edge_pixels)
        queue   = deque([floodfill_seed])

        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for dx_, dy_ in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx_, y + dy_
                if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in visited:
                    queue.append((nx, ny))

        mountain_area = {
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in visited
        }

        self.apply_terrain_to_grid(mountain_area, label)

import os
from PIL import Image
import numpy as np
import pandas as pd
import ast

from src.classification.labelers.stego_labeler import STEGOLabeler
from src.classification.labelers.lseg_labeler import LSegLabeler
from src.classification.labelers.clipseg_labeler import CLIPSegLabeler
from src.classification.labelers.labeler import Labeler
from src.classification.map_segmenter import MapSegmenter

def run_all_labelers(image_path, text_labels, object_labels, terrain_labels, grid_cell_size=16):
    image = Image.open(image_path).convert("RGB")

    stego_labeller = STEGOLabeler(
        image           = image,
        text_labels     = text_labels,
        model_ckpt_path = 'data/checkpoints/cocostuff27_vit_base_5.ckpt'
    )
    grid_stego = stego_labeller.segment_and_label_grid()
    print("STEGOLabeler grid:\n", grid_stego)

    lseg_labeller = LSegLabeler(
        image           = image,
        text_labels     = text_labels,
        model_ckpt_path = 'data/checkpoints/lseg_minimal_e200.ckpt'
    )
    grid_lseg = lseg_labeller.segment_and_label_grid()
    print("LSEGLabeler grid:\n", grid_lseg)

    map_results = {}

    ms_slic_no       = MapSegmenter(use_cell=False, use_adaptive=False, slic_growing=False)
    grid_map_slic_no = ms_slic_no.process_image(
        image_path,
        object_labels  = object_labels,
        terrain_labels = terrain_labels,
        grid_size      = grid_cell_size,
        n_segments     = [300, 100, 20],
        guided         = True
    )
    print("MapSegmenter SLIC no-growing grid:\n", grid_map_slic_no)
    map_results['map_slic_no'] = grid_map_slic_no


    ms_slic_grow       = MapSegmenter(use_cell=False, use_adaptive=False, slic_growing=True)
    grid_map_slic_grow = ms_slic_grow.process_image(
        image_path,
        object_labels  = object_labels,
        terrain_labels = terrain_labels,
        grid_size      = grid_cell_size,
        n_segments     = [300, 100, 20],
        guided         = True
    )
    print("MapSegmenter SLIC growing grid:\n", grid_map_slic_grow)
    map_results['map_slic_grow'] = grid_map_slic_grow

    ms_slic_adapt       = MapSegmenter(use_cell=False, use_adaptive=True)
    grid_map_slic_adapt = ms_slic_adapt.process_image(
        image_path,
        object_labels  = object_labels,
        terrain_labels = terrain_labels,
        grid_size      = grid_cell_size,
        n_segments     = [300, 100, 20],
        guided         = True
    )
    print("MapSegmenter SLIC adaptive grid:\n", grid_map_slic_adapt)
    map_results['map_slic_adapt'] = grid_map_slic_adapt

    ms_cell_no       = MapSegmenter(use_cell=True, use_adaptive=False)
    grid_map_cell_no = ms_cell_no.process_image(
        image_path,
        object_labels  = object_labels,
        terrain_labels = terrain_labels,
        grid_size      = grid_cell_size,
        n_segments     = [300, 100, 20],
        guided         = True
    )
    print("MapSegmenter Cell normal grid:\n", grid_map_cell_no)
    map_results['map_cell_no'] = grid_map_cell_no

    ms_cell_adapt       = MapSegmenter(use_cell=True, use_adaptive=True)
    grid_map_cell_adapt = ms_cell_adapt.process_image(
        image_path,
        object_labels  = object_labels,
        terrain_labels = terrain_labels,
        grid_size      = grid_cell_size,
        n_segments     = [300, 100, 20],
        guided         = True
    )
    print("MapSegmenter Cell adaptive grid:\n", grid_map_cell_adapt)
    map_results['map_cell_adapt'] = grid_map_cell_adapt

    clip_labeller = CLIPSegLabeler(
        image,
        text_labels    = text_labels,
        object_labels  = object_labels,
        terrain_labels = terrain_labels
    )
    grid_clipseg = clip_labeller.segment_and_label_grid()
    print("CLIPSegLabeler grid:\n", grid_clipseg)

    metrics = {
        'stego'  : grid_stego,
        'lseg'   : grid_lseg,
        'clipseg': grid_clipseg,
        **map_results
    }

    smoothed_metrics = {}
    for name, grid in metrics.items(): 
        try : 
            smoothed_grid                   = Labeler.smooth_grid(grid, text_labels)
            smoothed_metrics[name + '_crf'] = smoothed_grid
            print(f"CRF-smoothed grid for {name}:\n", smoothed_grid)
        except Exception as e: 
            print(f"CRF smoothing failed for {name} with error: {e}")
            smoothed_metrics[name + '_crf'] = None

    metrics.update(smoothed_metrics)
    return metrics


if __name__    == '__main__':
   csv_path     = os.getenv('DATA_CSV', 'data/output/image_objects.csv')
   df           = pd.read_csv(csv_path)
   labels_dir   = os.getenv('LABELS_DIR', 'data/output/labels')
   all_results  = []

    for root,                          _, files in os.walk(labels_dir): 
        for f in files                   : 
            if  not f.endswith('_labels.npy'): 
                continue
            label_path = os.path.join(root, f)
            image_name = f.replace('_labels.npy', '')
            rel_subdir = os.path.relpath(root, labels_dir)
            csv_match  = df[df['path'].str.endswith(os.path.join(rel_subdir, image_name + '.jpg'))]
            if csv_match.empty: 
                continue
            row = csv_match.iloc[0]

            image_path     = row['path'].replace('../../', '')
            gt_grid        = np.load(label_path, allow_pickle=True)
            object_labels  = [x.strip() for x in ast.literal_eval(row['objects'])]
            terrain_labels = [x.strip() for x in ast.literal_eval(row['terrains'])]
            text_labels    = object_labels + terrain_labels

            metrics = run_all_labelers(
                image_path,
                text_labels,
                object_labels,
                terrain_labels
            )

            scores = {}
            for name, pred_grid in metrics.items(): 
                if  pred_grid is not None and pred_grid.shape == gt_grid.shape: 
                    scores[name + '_acc'] = np.mean(pred_grid == gt_grid)
                else: 
                    scores[name + '_acc'] = None

            all_results.append({'image': image_path, **scores})

    results_df = pd.DataFrame(all_results)
    out_csv    = os.getenv('METRICS_OUT', 'data/output/all_metrics_scores.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    print(f"Saved accuracy metrics to {out_csv}")

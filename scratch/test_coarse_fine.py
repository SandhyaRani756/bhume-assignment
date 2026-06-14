import sys
from pathlib import Path
import time
import numpy as np
import geopandas as gpd
from scipy.ndimage import distance_transform_edt
from shapely.affinity import translate
from shapely.geometry import Polygon, MultiPolygon

# Adjust path to import bhume
sys.path.append(str(Path(r"c:\Users\USER\OneDrive\Desktop\BhuMi_Assignment\bhume-starter-kit\bhume-starter-kit").absolute()))

from bhume import load, score
from bhume.geo import open_imagery

def geom_to_points(geom, sample_dist=2.0):
    points = []
    if isinstance(geom, Polygon):
        ext = geom.exterior
        length = ext.length
        dists = np.arange(0, length, sample_dist)
        for d in dists:
            pt = ext.interpolate(d)
            points.append((pt.x, pt.y))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            ext = poly.exterior
            length = ext.length
            dists = np.arange(0, length, sample_dist)
            for d in dists:
                pt = ext.interpolate(d)
                points.append((pt.x, pt.y))
    return np.array(points) if points else np.empty((0, 2))

def evaluate_translations(points, grid, dt, tc, ta, tf, te, width, height):
    # Shape of points: (N, 2)
    # Shape of grid: (M, 2)
    trans_coords = points[:, None, :] + grid[None, :, :] # (N, M, 2)
    cols = (trans_coords[..., 0] - tc) / ta
    rows = (trans_coords[..., 1] - tf) / te
    cols_idx = np.clip(np.round(cols).astype(int), 0, width - 1)
    rows_idx = np.clip(np.round(rows).astype(int), 0, height - 1)
    dists = dt[rows_idx, cols_idx]
    return np.mean(dists, axis=0) # (M,)

def main():
    village_dir = r"c:\Users\USER\OneDrive\Desktop\BhuMi_Assignment\bhume-starter-kit\bhume-starter-kit\data\34855_vadnerbhairav_chandavad_nashik"
    village = load(village_dir)
    
    print("Loading boundaries raster...")
    with open_imagery(village.boundaries_path) as src:
        bounds_data = src.read(1)
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs
        
    print("Computing EDT...")
    t0 = time.time()
    dt = distance_transform_edt(bounds_data == 0)
    print(f"EDT computed in {time.time() - t0:.3f}s")
    
    plots_3857 = village.plots.to_crs(crs)
    truths_3857 = village.example_truths.to_crs(crs)
    
    # Coarse grid: -20 to 20 with step 2.0m
    coarse_range = np.arange(-20.0, 20.1, 2.0)
    c_dx, c_dy = np.meshgrid(coarse_range, coarse_range)
    coarse_grid = np.stack([c_dx.flatten(), c_dy.flatten()], axis=1)
    
    # Fine grid: -2.0 to 2.0 with step 0.5m
    fine_range = np.arange(-2.0, 2.01, 0.5)
    f_dx, f_dy = np.meshgrid(fine_range, fine_range)
    fine_grid = np.stack([f_dx.flatten(), f_dy.flatten()], axis=1)
    
    aligned_geoms = {}
    
    tc = transform.c
    ta = transform.a
    tf = transform.f
    te = transform.e
    
    t0 = time.time()
    for pn in truths_3857.index:
        geom = plots_3857.loc[pn, 'geometry']
        points = geom_to_points(geom, sample_dist=2.0)
        if len(points) == 0:
            aligned_geoms[pn] = geom
            continue
            
        # 1. Coarse search
        coarse_dists = evaluate_translations(points, coarse_grid, dt, tc, ta, tf, te, width, height)
        best_coarse_idx = np.argmin(coarse_dists)
        best_c_dx, best_c_dy = coarse_grid[best_coarse_idx]
        
        # 2. Fine search around best coarse offset
        local_fine_grid = fine_grid + np.array([best_c_dx, best_c_dy])
        fine_dists = evaluate_translations(points, local_fine_grid, dt, tc, ta, tf, te, width, height)
        best_fine_idx = np.argmin(fine_dists)
        best_dx, best_dy = local_fine_grid[best_fine_idx]
        best_dist = fine_dists[best_fine_idx]
        
        print(f"Plot {pn}: Coarse ({best_c_dx:.1f}, {best_c_dy:.1f}) -> Fine dx={best_dx:.2f}m, dy={best_dy:.2f}m | dist: {best_dist:.2f}")
        
        aligned_geoms[pn] = translate(geom, best_dx, best_dy)
        
    print(f"Finished search in {time.time() - t0:.3f}s")
    
    # Let's see the score
    preds_local = village.plots.copy()
    preds_local['status'] = 'corrected'
    preds_local['confidence'] = 0.8
    preds_local['method_note'] = 'coarse-fine local search'
    
    for pn, g in aligned_geoms.items():
        g_4326 = gpd.GeoSeries([g], crs=crs).to_crs('EPSG:4326').iloc[0]
        preds_local.loc[pn, 'geometry'] = g_4326
        
    score_local = score(preds_local, village)
    print("\n--- Coarse-Fine Locally Aligned Score ---")
    print(score_local)

if __name__ == '__main__':
    main()

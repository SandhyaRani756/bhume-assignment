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
    
    # Estimate global shift (in EPSG:3857) to know what to expect
    dxs, dys = [], []
    for pn in truths_3857.index:
        if pn in plots_3857.index:
            o = plots_3857.loc[pn, 'geometry'].centroid
            t = truths_3857.loc[pn, 'geometry'].centroid
            dxs.append(t.x - o.x)
            dys.append(t.y - o.y)
    mdx, mdy = np.median(dxs), np.median(dys)
    print(f"Global shift: dx={mdx:.2f}m, dy={mdy:.2f}m")
    
    # Search grid relative to OFFICIAL geometry
    # Since global shift is about (-4.7, 12.1), we want to make sure the grid covers:
    # - official (0, 0)
    # - global shift (-4.7, 12.1)
    # So dx in [-15, 10], dy in [-10, 20] is good. Or just a symmetric [-20, 20] grid.
    search_range = np.arange(-20.0, 20.1, 1.0)
    grid_dx, grid_dy = np.meshgrid(search_range, search_range)
    grid_dx = grid_dx.flatten()
    grid_dy = grid_dy.flatten()
    grid = np.stack([grid_dx, grid_dy], axis=1) # shape (M, 2)
    print(f"Grid size: {len(grid)} offsets")
    
    aligned_geoms = {}
    
    tc = transform.c
    ta = transform.a
    tf = transform.f
    te = transform.e
    
    t0 = time.time()
    for pn in truths_3857.index:
        geom = plots_3857.loc[pn, 'geometry'] # OFFICIAL, not shifted!
        points = geom_to_points(geom, sample_dist=2.0)
        if len(points) == 0:
            aligned_geoms[pn] = geom
            continue
            
        trans_coords = points[:, None, :] + grid[None, :, :]
        cols = (trans_coords[..., 0] - tc) / ta
        rows = (trans_coords[..., 1] - tf) / te
        cols_idx = np.clip(np.round(cols).astype(int), 0, width - 1)
        rows_idx = np.clip(np.round(rows).astype(int), 0, height - 1)
        
        dists = dt[rows_idx, cols_idx]
        avg_dists = np.mean(dists, axis=0)
        
        best_idx = np.argmin(avg_dists)
        best_dx, best_dy = grid[best_idx]
        best_dist = avg_dists[best_idx]
        
        print(f"Plot {pn} (from official): Best translation dx={best_dx:.2f}m, dy={best_dy:.2f}m | dist: {avg_dists[len(grid)//2]:.2f} -> {best_dist:.2f}")
        
        aligned_geoms[pn] = translate(geom, best_dx, best_dy)
        
    print(f"Finished search in {time.time() - t0:.3f}s")
    
    # Let's see the score
    preds_local = village.plots.copy()
    preds_local['status'] = 'corrected'
    preds_local['confidence'] = 0.8
    preds_local['method_note'] = 'local search'
    
    for pn, g in aligned_geoms.items():
        g_4326 = gpd.GeoSeries([g], crs=crs).to_crs('EPSG:4326').iloc[0]
        preds_local.loc[pn, 'geometry'] = g_4326
        
    score_local = score(preds_local, village)
    print("\n--- Locally Aligned Score (from official) ---")
    print(score_local)

if __name__ == '__main__':
    main()

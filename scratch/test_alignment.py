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
    """Extract boundary points from a geometry."""
    points = []
    if isinstance(geom, Polygon):
        ext = geom.exterior
        # sample along the exterior
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
    
    # 1. Load boundaries raster and compute EDT
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
    
    # Reproject plots and example truths to boundaries CRS (EPSG:3857)
    plots_3857 = village.plots.to_crs(crs)
    truths_3857 = village.example_truths.to_crs(crs)
    
    # 2. Estimate global shift (in EPSG:3857)
    dxs, dys = [], []
    for pn in truths_3857.index:
        if pn in plots_3857.index:
            o = plots_3857.loc[pn, 'geometry'].centroid
            t = truths_3857.loc[pn, 'geometry'].centroid
            dxs.append(t.x - o.x)
            dys.append(t.y - o.y)
    mdx, mdy = np.median(dxs), np.median(dys)
    print(f"Global shift: dx={mdx:.2f}m, dy={mdy:.2f}m")
    
    # Apply global shift
    shifted_plots = plots_3857.copy()
    shifted_plots['geometry'] = shifted_plots.geometry.apply(lambda g: translate(g, mdx, mdy))
    
    # Define local search grid
    # Search in a grid: dx and dy from -15 to 15 meters with step 1.0 meters
    search_range = np.arange(-15.0, 15.1, 1.0)
    grid_dx, grid_dy = np.meshgrid(search_range, search_range)
    grid_dx = grid_dx.flatten()
    grid_dy = grid_dy.flatten()
    grid = np.stack([grid_dx, grid_dy], axis=1) # shape (M, 2)
    print(f"Local grid size: {len(grid)} offsets")
    
    # 3. For each plot in the example truths, perform local search
    print("\nAligning example truths locally...")
    aligned_geoms = {}
    
    # Let's unpack transform components
    tc = transform.c
    ta = transform.a
    tf = transform.f
    te = transform.e
    
    for pn in truths_3857.index:
        geom = shifted_plots.loc[pn, 'geometry']
        points = geom_to_points(geom, sample_dist=2.0) # shape (N, 2)
        if len(points) == 0:
            aligned_geoms[pn] = geom
            continue
            
        # Compute all translated coordinates: (N, M, 2)
        trans_coords = points[:, None, :] + grid[None, :, :]
        
        # Convert to pixel coords
        cols = (trans_coords[..., 0] - tc) / ta
        rows = (trans_coords[..., 1] - tf) / te
        
        # Round and clip
        cols_idx = np.clip(np.round(cols).astype(int), 0, width - 1)
        rows_idx = np.clip(np.round(rows).astype(int), 0, height - 1)
        
        # Look up distance transform values: (N, M)
        dists = dt[rows_idx, cols_idx]
        
        # Average distance for each translation: (M,)
        avg_dists = np.mean(dists, axis=0)
        
        # Find best translation
        best_idx = np.argmin(avg_dists)
        best_dx, best_dy = grid[best_idx]
        best_dist = avg_dists[best_idx]
        
        # Print results
        orig_idx = np.where((grid[:, 0] == 0) & (grid[:, 1] == 0))[0][0]
        orig_dist = avg_dists[orig_idx]
        print(f"Plot {pn}: Best translation dx={best_dx:.2f}m, dy={best_dy:.2f}m | dist: {orig_dist:.2f} -> {best_dist:.2f}")
        
        # Store aligned geom
        aligned_geoms[pn] = translate(geom, best_dx, best_dy)
        
    # Let's construct predictions gdf for self-scoring
    # We will score:
    # 1. Global shift predictions
    # 2. Locally aligned predictions (for truths only, and others global shift)
    
    # Global predictions
    preds_global = village.plots.copy()
    preds_global['geometry'] = preds_global.to_crs(crs).geometry.apply(lambda g: translate(g, mdx, mdy)).to_crs('EPSG:4326')
    preds_global['status'] = 'corrected'
    preds_global['confidence'] = 0.8
    preds_global['method_note'] = 'global shift'
    
    score_global = score(preds_global, village)
    print("\n--- Global Shift Score ---")
    print(score_global)
    
    # Locally aligned predictions
    preds_local = preds_global.copy()
    for pn, g in aligned_geoms.items():
        # Convert back to EPSG:4326
        g_4326 = gpd.GeoSeries([g], crs=crs).to_crs('EPSG:4326').iloc[0]
        preds_local.loc[pn, 'geometry'] = g_4326
        preds_local.loc[pn, 'confidence'] = 0.95
        preds_local.loc[pn, 'method_note'] = 'local aligned'
        
    score_local = score(preds_local, village)
    print("\n--- Locally Aligned Score (on truths) ---")
    print(score_local)

if __name__ == '__main__':
    main()

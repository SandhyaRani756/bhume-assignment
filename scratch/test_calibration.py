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
    trans_coords = points[:, None, :] + grid[None, :, :]
    cols = (trans_coords[..., 0] - tc) / ta
    rows = (trans_coords[..., 1] - tf) / te
    cols_idx = np.clip(np.round(cols).astype(int), 0, width - 1)
    rows_idx = np.clip(np.round(rows).astype(int), 0, height - 1)
    dists = dt[rows_idx, cols_idx]
    return np.mean(dists, axis=0)

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
    dt = distance_transform_edt(bounds_data == 0)
    
    plots_3857 = village.plots.to_crs(crs)
    truths_3857 = village.example_truths.to_crs(crs)
    
    # Estimate global shift
    dxs, dys = [], []
    for pn in truths_3857.index:
        if pn in plots_3857.index:
            o = plots_3857.loc[pn, 'geometry'].centroid
            t = truths_3857.loc[pn, 'geometry'].centroid
            dxs.append(t.x - o.x)
            dys.append(t.y - o.y)
    mdx, mdy = np.median(dxs), np.median(dys)
    print(f"Global shift: dx={mdx:.2f}m, dy={mdy:.2f}m")
    
    g_rx = int(round(mdx))
    g_ry = int(round(mdy))
    
    # Bounding box covering (0,0) and (g_rx, g_ry) with padding 15m
    dx_min = min(0, g_rx) - 15
    dx_max = max(0, g_rx) + 15
    dy_min = min(0, g_ry) - 15
    dy_max = max(0, g_ry) + 15
    
    search_dx = np.arange(dx_min, dx_max + 1, 1.0)
    search_dy = np.arange(dy_min, dy_max + 1, 1.0)
    grid_dx, grid_dy = np.meshgrid(search_dx, search_dy)
    grid = np.stack([grid_dx.flatten(), grid_dy.flatten()], axis=1)
    print(f"Grid size: {len(grid)} offsets")
    
    aligned_geoms = {}
    confidences = {}
    statuses = {}
    
    tc = transform.c
    ta = transform.a
    tf = transform.f
    te = transform.e
    
    # We will score on truths
    for pn in truths_3857.index:
        geom = plots_3857.loc[pn, 'geometry']
        points = geom_to_points(geom, sample_dist=2.0)
        if len(points) == 0:
            statuses[pn] = 'flagged'
            aligned_geoms[pn] = geom
            confidences[pn] = 0.0
            continue
            
        avg_dists = evaluate_translations(points, grid, dt, tc, ta, tf, te, width, height)
        best_idx = np.argmin(avg_dists)
        best_dx, best_dy = grid[best_idx]
        best_dist = avg_dists[best_idx]
        
        # Determine status and confidence
        # If best_dist is large, e.g. > 4.0 pixels, we are not confident, so flag it
        pixel_res = abs(ta) # resolution in meters per pixel (about 2.39)
        best_dist_m = best_dist * pixel_res
        
        # Distance to anchors: official (0,0) and global shift (mdx, mdy)
        dist_to_official = np.sqrt(best_dx**2 + best_dy**2)
        dist_to_global = np.sqrt((best_dx - mdx)**2 + (best_dy - mdy)**2)
        min_anchor_dist = min(dist_to_official, dist_to_global)
        
        # Compute a raw confidence score
        # Let's say confidence is higher if best_dist_m is small
        # And we penalize if it's far from the global shift / official
        conf = 1.0 / (1.0 + best_dist_m / 3.0) # e.g. if best_dist_m is 0, conf is 1.0. If 3m, conf is 0.5.
        
        # Penalize if it's far from anchors
        if min_anchor_dist > 5.0:
            # penalize confidence proportionally
            conf *= np.exp(-(min_anchor_dist - 5.0) / 10.0)
            
        # Threshold for flagging
        # If best_dist is too large (e.g. > 3.5 pixels, which is about 8.4 meters), flag it
        # Also flag if confidence is very low
        if best_dist > 3.5 or conf < 0.35:
            status = 'flagged'
            geom_pred = geom # keep original
        else:
            status = 'corrected'
            geom_pred = translate(geom, best_dx, best_dy)
            
        statuses[pn] = status
        aligned_geoms[pn] = geom_pred
        confidences[pn] = float(np.clip(conf, 0.0, 1.0))
        
        print(f"Plot {pn}: Best translation dx={best_dx:.2f}m, dy={best_dy:.2f}m | dist_m: {best_dist_m:.2f}m | anchor_dist: {min_anchor_dist:.2f}m | conf: {confidences[pn]:.3f} | status: {status}")
        
    # Let's construct predictions gdf for self-scoring
    preds_local = village.plots.copy()
    preds_local['status'] = 'corrected'
    preds_local['confidence'] = 0.8
    preds_local['method_note'] = 'local search'
    
    for pn in truths_3857.index:
        g_4326 = gpd.GeoSeries([aligned_geoms[pn]], crs=crs).to_crs('EPSG:4326').iloc[0]
        preds_local.loc[pn, 'geometry'] = g_4326
        preds_local.loc[pn, 'status'] = statuses[pn]
        preds_local.loc[pn, 'confidence'] = confidences[pn]
        preds_local.loc[pn, 'method_note'] = f"aligned dx={preds_local.loc[pn].geometry.centroid.x - village.plots.loc[pn].geometry.centroid.x:.2f} dy={preds_local.loc[pn].geometry.centroid.y - village.plots.loc[pn].geometry.centroid.y:.2f}"

    score_local = score(preds_local, village)
    print("\n--- Calibration and Restraint Score ---")
    print(score_local)

if __name__ == '__main__':
    main()

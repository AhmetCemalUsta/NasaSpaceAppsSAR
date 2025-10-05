# run_all.py  — Miniforge ortamında tek dosya uçtan uca çalıştır
# Çalıştırma:  conda activate C:\Users\cemal\wildfire-local\env  →  python -u run_all.py
import os, sys, math, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import date, timedelta
from shapely.geometry import shape, box, Point
from shapely.ops import transform
from shapely.prepared import prep
try:
    from shapely import make_valid
except Exception:
    from shapely.validation import make_valid
import fiona
from pyproj import CRS, Transformer

import xarray as xr
from pystac_client import Client
import stackstac
import rioxarray
import lightgbm as lgb

# -------------------- KONFİG --------------------
PROJECT = Path(r"C:\Users\cemal\wildfire-local")
DATA    = PROJECT / "data"
GLOBFIRE_SRC = Path(r"C:\Users\cemal\Desktop\GLOBFIRE_burned_area_full_dataset_2002_2023")  # klasör ya da tek .shp/.gpkg
RES_M  = 1000   # grid çözünürlüğü (m). 500 çok yavaş olabilir; 1000 dengeli.
TAU    = 7      # gün, ileri pencere
X_M2   = 1_000_000  # 100 ha
TRAIN_Y, TEST_Y = 2021, 2023
M_START, M_END = "07-01", "08-31"
ROI_LONLAT = (28.0, 36.0, 31.0, 37.8)  # (minx, miny, maxx, maxy)
EPSG_UTM = 32635
# ------------------------------------------------

DATA.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(msg, flush=True)

def ensure_roi():
    roi_gpkg = DATA / "roi_mugla_antalya.gpkg"
    if roi_gpkg.exists():
        return gpd.read_file(roi_gpkg)
    from shapely.geometry import box
    g = gpd.GeoDataFrame({"name":["ROI"]}, geometry=[box(*ROI_LONLAT)], crs="EPSG:4326").to_crs(EPSG_UTM)
    g.to_file(roi_gpkg, driver="GPKG")
    return g

def drop_z(geom):
    if geom.is_empty: return geom
    return transform(lambda x,y,z=None:(x,y), geom)

def finite_bounds(geom):
    xmin,ymin,xmax,ymax = geom.bounds
    return np.isfinite([xmin,ymin,xmax,ymax]).all()

def parse_date(props):
    for k,v in props.items():
        lk = k.lower()
        if any(s in lk for s in ["date","start","initial","rep"]):
            dt = pd.to_datetime(v, errors="coerce")
            if not pd.isna(dt): return dt
    return pd.NaT

def gather_sources(path: Path):
    if path.is_file(): return [path]
    if path.is_dir():  return list(path.rglob("*.shp")) + list(path.rglob("*.gpkg"))
    raise SystemExit(f"Kaynak bulunamadı: {path}")

def process_vector_file(path, roi_gdf):
    log(f"[OPEN] {path}")
    with fiona.open(str(path)) as src:
        n = len(src)
        if src.crs_wkt: crs_src = CRS.from_wkt(src.crs_wkt)
        elif src.crs:   crs_src = CRS.from_user_input(src.crs)
        else:           crs_src = CRS.from_epsg(4326)

        roi_src = roi_gdf.to_crs(crs_src).union_all().buffer(0)
        roi_src_prep = prep(roi_src)
        rbox = box(*roi_src.bounds)

        to_utm = Transformer.from_crs(crs_src, CRS.from_epsg(EPSG_UTM), always_xy=True).transform
        recs = []
        log(f"[INFO] features={n} | CRS={crs_src.to_string()}")
        for i, feat in enumerate(src):
            if i % 1000 == 0: log(f"[PROGRESS] {i}/{n}")
            gj = feat.get("geometry")
            if gj is None: continue
            try:
                g = shape(gj)
            except Exception:
                continue
            if not g.bounds or not box(*g.bounds).intersects(rbox): 
                continue
            try:
                if not roi_src_prep.intersects(g): 
                    continue
            except Exception:
                continue
            g = drop_z(g)
            try: g = make_valid(g)
            except Exception: pass
            try: g = g.buffer(0)
            except Exception: continue
            if g.is_empty or not finite_bounds(g): 
                continue
            try:
                gutm = transform(to_utm, g)
            except Exception:
                continue
            if gutm.is_empty or not finite_bounds(gutm): 
                continue
            dt = parse_date(feat.get("properties", {}))
            if pd.isna(dt): 
                continue
            recs.append({"date": dt, "geometry": gutm})
    if not recs:
        log("[WARN] dosyadan geçerli öğe yok")
        return gpd.GeoDataFrame(columns=["date","geometry"], geometry="geometry", crs=EPSG_UTM)
    g = gpd.GeoDataFrame(recs, geometry="geometry", crs=EPSG_UTM)
    log(f"[DONE] kept {len(g)}")
    return g

def build_perims():
    roi_gdf = ensure_roi()
    outs = []
    for src in gather_sources(GLOBFIRE_SRC):
        try:
            g = process_vector_file(src, roi_gdf)
            if not g.empty: outs.append(g)
        except Exception as e:
            log(f"[ERROR] {src}: {e}")
    if not outs: 
        raise SystemExit("Perimetre bulunamadı.")
    g = gpd.GeoDataFrame(pd.concat(outs, ignore_index=True), crs=EPSG_UTM)
    g["date"] = pd.to_datetime(g["date"])
    out_all = DATA / "perims_all.gpkg"
    g_out = g.copy(); g_out["date"] = g_out["date"].dt.date
    g_out.to_file(out_all, driver="GPKG")
    log(f"[WRITE] {out_all} rows={len(g_out)}")
    sels = []
    for yr in [TRAIN_Y, TEST_Y]:
        a = pd.Timestamp(f"{yr}-{M_START}"); b = pd.Timestamp(f"{yr}-{M_END}")
        sub = g[(g["date"]>=a) & (g["date"]<=b)].copy()
        sub["date"] = sub["date"].dt.date
        pth = DATA / f"perims_{yr}.gpkg"
        sub.to_file(pth, driver="GPKG")
        log(f"[WRITE] {pth} rows={len(sub)}")
        sels.append(sub)
    merged = gpd.GeoDataFrame(pd.concat(sels, ignore_index=True), crs=EPSG_UTM)
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    pth = DATA / "perims.gpkg"
    merged.to_file(pth, driver="GPKG")
    log(f"[WRITE] {pth} rows={len(merged)}")
    return pth

def build_grid(roi_utm, res=RES_M):
    minx,miny,maxx,maxy = roi_utm.bounds
    xs = np.arange(minx, maxx, res); ys = np.arange(miny, maxy, res)
    tiles=[]; ids=[]; k=0
    for x in xs:
        for y in ys:
            g=box(x,y,x+res,y+res)
            if g.intersects(roi_utm): 
                tiles.append(g); ids.append(k); k+=1
    grid=gpd.GeoDataFrame({"tile_id":ids}, geometry=tiles, crs=EPSG_UTM)
    return grid

def make_labels(perims_path):
    ROI = gpd.read_file(DATA / "roi_mugla_antalya.gpkg").to_crs(EPSG_UTM).union_all()
    PERIMS = gpd.read_file(perims_path).to_crs(EPSG_UTM)
    PERIMS["date"] = pd.to_datetime(PERIMS["date"]).dt.date
    grid = build_grid(ROI, RES_M)
    log(f"[GRID] {len(grid)} hücre @ {RES_M} m")
    def run_period(y, out_csv):
        start = date(y,7,1); end = date(y,9,1)
        days = pd.date_range(start, end, freq="D").date
        rows=[]
        for i,d in enumerate(days):
            if i%5==0: log(f"[LABEL {y}] {i+1}/{len(days)} {d}")
            per_w = PERIMS[(PERIMS["date"]>=d) & (PERIMS["date"]<= d+timedelta(days=TAU))]
            if len(per_w):
                u = per_w.unary_union
                ia = grid.geometry.intersection(u).area
                ylab = (ia >= X_M2).astype(int)
            else:
                ylab = pd.Series(0, index=grid.index)
            past = PERIMS[PERIMS["date"]<=d]
            if len(past):
                sidx = past.sindex; tsb=[]
                for g in grid.geometry:
                    cand=list(sidx.intersection(g.bounds))
                    pp=past.iloc[cand]; pp=pp[pp.intersects(g)]
                    if len(pp)==0: tsb.append(9999)
                    else:
                        last=pp.sort_values("date",ascending=False).iloc[0]["date"]
                        tsb.append((d-last).days)
            else:
                tsb=[9999]*len(grid)
            rows.append(pd.DataFrame({"tile_id":grid.tile_id,"date":d,"y":ylab.values,"time_since_burn":tsb}))
        df=pd.concat(rows,ignore_index=True)
        df.to_csv(PROJECT/out_csv,index=False)
        log(f"[WRITE] {PROJECT/out_csv} rows={len(df)}")
    run_period(TRAIN_Y, f"labels_{TRAIN_Y}_JulAug.csv")
    run_period(TEST_Y,  f"labels_{TEST_Y}_JulAug.csv")

def fetch_stack(BBOX_lonlat, start, end):
    api = Client.open("https://earth-search.aws.element84.com/v1")
    it = api.search(collections=["sentinel-1-rtc"], bbox=BBOX_lonlat, datetime=f"{start}/{end}",
                    query={"sar:instrument_mode":{"eq":"IW"}, "polarizations":{"all":["VV","VH"]}})
    items = list(it.get_items())
    if not items: 
        return None
    da = stackstac.stack(items, assets=["sigma0_vv","sigma0_vh"], resolution=30, bounds_latlon=BBOX_lonlat, epsg=EPSG_UTM, chunks={"time":-1})
    da = 10*np.log10(da.clip(min=1e-6)).rename({"asset":"band"}).assign_coords(band=["VV","VH"])
    return da

def slope(arr):
    t = xr.DataArray(np.arange(arr.sizes["time"]), dims=["time"])
    xm = t.mean().item(); ym = arr.mean("time")
    return (((t-xm)*(arr-ym)).sum("time"))/(((t-xm)**2).sum("time"))

def feats_for_day(ds14, ds30):
    vv14 = ds14.sel(band="VV").mean("time"); vh14 = ds14.sel(band="VH").mean("time"); r14 = vh14 - vv14
    vv30 = ds30.sel(band="VV"); vh30 = ds30.sel(band="VH")
    vvmean = vv30.mean("time"); vhmean = vh30.mean("time")
    vvstd = vv30.std("time").fillna(1.0); vhstd = vh30.std("time").fillna(1.0)
    vv_z30 = (vvmean - vvmean.mean())/vvstd
    vh_z30 = (vhmean - vhmean.mean())/vhstd
    r_z30  = ((vhmean - vvmean) - (vhmean - vvmean).mean())/((vhstd**2+vvstd**2)**0.5)
    vv_slope30 = slope(vv30); vh_slope30 = slope(vh30)
    return xr.Dataset({"VV_14d":vv14,"VH_14d":vh14,"R_14d":r14,"VV_z30":vv_z30,"VH_z30":vh_z30,"R_z30":r_z30,"VV_slope30":vv_slope30,"VH_slope30":vh_slope30})

def sample_grid_centroids(ds, grid):
    ds = ds.rio.write_crs(EPSG_UTM)
    cents = grid.geometry.centroid
    xs = xr.DataArray(cents.x.values, dims="points")
    ys = xr.DataArray(cents.y.values, dims="points")
    arr = ds.to_array()  # [band, y, x]
    vals = arr.sel(x=xs, y=ys, method="nearest").to_series().unstack(0)
    vals["tile_id"] = grid.tile_id.values
    return vals.reset_index(drop=True)

def build_s1_features():
    roi_ll = gpd.GeoSeries([gpd.read_file(DATA/"roi_mugla_antalya.gpkg").to_crs(4326).union_all()]).total_bounds
    roi_utm = gpd.read_file(DATA/"roi_mugla_antalya.gpkg").to_crs(EPSG_UTM).union_all()
    grid = build_grid(roi_utm, RES_M)
    def run_period(y, out_csv):
        start = pd.Timestamp(f"{y}-07-01"); end = pd.Timestamp(f"{y}-09-01")
        days = pd.date_range(start, end, freq="D")
        out=[]
        for i,d in enumerate(days):
            log(f"[S1 {y}] {i+1}/{len(days)} {d.date()}")
            d0 = (d - pd.Timedelta(days=30)).date().isoformat()
            d1 = (d - pd.Timedelta(days=14)).date().isoformat()
            ds30 = fetch_stack(roi_ll, d0, d.date().isoformat())
            ds14 = fetch_stack(roi_ll, d1, d.date().isoformat())
            if ds30 is None or ds14 is None: 
                continue
            feats = feats_for_day(ds14, ds30)
            df = sample_grid_centroids(feats, grid)
            df["date"]=d.date().isoformat()
            out.append(df)
        if out:
            res=pd.concat(out, ignore_index=True)
            res.to_csv(PROJECT/out_csv, index=False)
            log(f"[WRITE] {PROJECT/out_csv} rows={len(res)}")
    run_period(TRAIN_Y, f"features_s1_{TRAIN_Y}.csv")
    run_period(TEST_Y,  f"features_s1_{TEST_Y}.csv")

def train_and_predict():
    train = (pd.read_csv(PROJECT/f"features_s1_{TRAIN_Y}.csv")
             .merge(pd.read_csv(PROJECT/f"labels_{TRAIN_Y}_JulAug.csv"), on=["tile_id","date"], how="inner"))
    test  = (pd.read_csv(PROJECT/f"features_s1_{TEST_Y}.csv")
             .merge(pd.read_csv(PROJECT/f"labels_{TEST_Y}_JulAug.csv"), on=["tile_id","date"], how="inner"))
    feats = ["VV_14d","VH_14d","R_14d","VV_z30","VH_z30","R_z30","VV_slope30","VH_slope30","time_since_burn"]
    train=train.dropna(subset=feats+["y"])
    test =test.dropna(subset=feats)
    log(f"[ML] train={len(train)} test={len(test)} features={len(feats)}")
    m = lgb.LGBMClassifier(n_estimators=1500,learning_rate=0.03,num_leaves=63,subsample=0.8,colsample_bytree=0.8,reg_lambda=2)
    m.fit(train[feats], train["y"].astype(int))
    test["prob"] = m.predict_proba(test[feats])[:,1]
    out = PROJECT/"pred_2023.csv"
    test[["tile_id","date","prob"]].to_csv(out, index=False)
    log(f"[WRITE] {out} rows={len(test)}")

def main():
    log("[STEP] ROI")
    ensure_roi()
    log("[STEP] PERIMS")
    perims_path = build_perims()
    log("[STEP] LABELS")
    make_labels(perims_path)
    log("[STEP] S1 FEATURES")
    build_s1_features()
    log("[STEP] TRAIN/PREDICT")
    train_and_predict()
    log("[OK] Tüm boru hattı bitti.")

if __name__ == "__main__":
    main()

from __future__ import annotations
import os, sys, math, datetime as dt, requests, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from datetime import datetime, timedelta, timezone
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.iokit import parse_tle
from skyfield.framelib import itrs
from tqdm import tqdm
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from io import StringIO # for processing text TLE data

# -------------------------- SUPPRESS WARNINGS FOR CLEAN OUTPUT --------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# -------------------------- CONFIG --------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
utc = timezone.utc
center = datetime(2025, 11, 18, 0, 0, tzinfo=utc) 
START_UTC = center - timedelta(hours=6) 
END_UTC = center + timedelta(hours=6) 
STEP_SEC = 60 
SAMPLE_STRIDE = 1 
OBS_LAT_DEG, OBS_LON_DEG, OBS_H_M = 42.3611, -71.0570, 10.0 
INCLUDE_GPS = INCLUDE_GALILEO = INCLUDE_GLONASS = True
LEO_GROUPS = ["starlink", "oneweb"]
ELEV_MASK_DEG = 0.0
GNSS_FREQ_HZ = {"GPS_L1": 1_575_420_000.0}
FREQ = GNSS_FREQ_HZ["GPS_L1"]
H_IONO_M = 450e3
LEO_CAP_MIN_M, LEO_CAP_MAX_M = 100e3, 2_000e3
TLE_SOURCE = 1
OUT_DIR = "out_leo_meo_iri"
os.makedirs(OUT_DIR, exist_ok=True)
R_E = 6_371_000.0
KAPPA = 40.308193
C = 299_792_458.0
DATE_LABEL = f"{center:%b %d, %Y}" 

# -------------------------- STARTUP FUNCTIONS --------------------------
def window_label_utc(s, e): return f"{s:%Y-%m-%d %H:%M}–{e:%Y-%m-%d %H:%M} UTC"
def window_stamp_utc(s, e): return f"{s:%Y%m%dT%H%M}Z_{e:%Y%m%dT%H%M}Z"

def obliquity_factor(elev_rad, h_iono_m=H_IONO_M):
    cE = math.cos(elev_rad)
    arg = 1.0 - ((R_E * cE) / (R_E + h_iono_m))**2
    return 1.0 / math.sqrt(max(arg, 1e-12))

def stec_from_vtec(vtec_tecu, elev_rad, h_iono_m=H_IONO_M):
    # TECU = 1e16 electrons/m^2
    return vtec_tecu * 1e16 * obliquity_factor(elev_rad, h_iono_m)

def code_group_delay_m(stec, f_hz): 
    # group delay (meters) = KAPPA * STEC / f^2
    return KAPPA * stec / (f_hz * f_hz)

def leo_cap_fraction(alt_m, hmin=LEO_CAP_MIN_M, hmax=LEO_CAP_MAX_M):
    return float(np.clip((alt_m - hmin) / (hmax - hmin), 0.0, 1.0))

def vtec_diurnal(t_utc, vtec_max=20.0, vtec_min=5.0, t_peak_local=14.0):
    hour = t_utc.hour + t_utc.minute/60.0
    # longitude converted to hours (15 deg = 1 hour)
    local_h = (hour + OBS_LON_DEG/15.0) % 24.0
    cos_term = np.cos(2*np.pi*(local_h - t_peak_local)/24.0)
    return vtec_min + (vtec_max-vtec_min)*(1.0+cos_term)/2.0

TLE_FILENAME = "daily_elsets_2025322.txt"

try:
    with open(TLE_FILENAME, 'rb') as f: 
        raw_content = f.read()
        HISTORICAL_TLE_FILE_CONTENT = raw_content.decode('utf-8') # decode bytes to string
    print(f"[INFO] Loaded TLE data from file: {TLE_FILENAME}")
except FileNotFoundError:
    print(f"[FATAL] Error: TLE file '{TLE_FILENAME}' not found. Exiting.")
    sys.exit(1)

# -------------------------- TLE PARSER --------------------------
def parse_tle_text_file(text: str) -> List[Tuple[str,str,str]]:
    """
    parses the TLE file content generating a name from the NORAD ID 
    and Epoch data for each satellite
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out, i = [], 0
    
    while i < len(lines):
        line1 = lines[i]
        
        if line1.startswith('1 '):
            if i + 1 < len(lines) and lines[i+1].startswith('2 '):
                line2 = lines[i+1]
                
                try:
                    # extract the NORAD ID (columns 3-7) and epoch year/day
                    norad_id = line1[2:7].strip()
                    epoch_year_day = line1[18:32].strip() 
                    
                    name = f"NORAD-{norad_id}-E{epoch_year_day}"
                    
                    out.append((name, line1, line2))
                    i += 2 
                except IndexError:
                    i += 1 
                    
            else:
                i += 1 
                
        else:
            i += 1 
            
    return out

def make_satellites_from_tles(triples, ts):
    return [EarthSatellite(l1, l2, name=name, ts=ts) for name, l1, l2 in triples]

# -------------------------- SKYFIELD --------------------------
def compute_time_grid(start_utc, end_utc, step_sec, ts):
    times = pd.date_range(start_utc, end_utc, freq=f"{step_sec}s", inclusive="left", tz="UTC")
    t_sky = ts.utc([t.year for t in times], [t.month for t in times], [t.day for t in times],
                   [t.hour for t in times], [t.minute for t in times], [t.second for t in times])
    return times, t_sky

def satellite_altitude_m(sat, t):
    sp = wgs84.subpoint(sat.at(t))
    return sp.elevation.m

def constellation_name(s: EarthSatellite) -> str:
    nm = (s.name or "").lower()
    if "gps" in nm or "norad-25338" in nm: return "GPS" # GPS-IIR-1
    if "galileo" in nm or "norad-43564" in nm: return "Galileo" # GALILEO-FOC-FM17
    if "glonass" in nm or "cosmos" in nm or "norad-45358" in nm: return "GLONASS" # GLONASS-M 752
    if "oneweb" in nm or "norad-48388" in nm: return "OneWeb" # Oneweb 148
    if "starlink" in nm or "norad-44713" in nm: return "Starlink" # Starlink 1007
    
    norad_id_prefix = s.name.split('-')[1] if s.name and s.name.startswith('NORAD-') else ''
    
    return "Other" 
    
# -------------------------- IRI (vectorized) -----------------
def hardcoded_ne_vec(lat_deg, lon_deg, h_m, t_utc):
    h_km = h_m / 1000.0
    vtec = vtec_diurnal(t_utc, vtec_max=30.0, vtec_min=5.0)
    # Chapman layer parameters
    hmF2 = 350.0
    HmF2 = 100.0
    NmF2 = vtec * 1e16 / (HmF2 * 1000.0 * np.sqrt(2 * np.pi))
    z = (h_km - hmF2) / HmF2
    Ne = NmF2 * np.exp(0.5 * (1 - z - np.exp(-z)))
    mask = (h_km >= 80) & (h_km <= 1500)
    return np.where(mask, Ne, 0.0)

def ecef_to_spherical_geodetic_vec(ecef_pts):
    x, y, z = ecef_pts[:, 0], ecef_pts[:, 1], ecef_pts[:, 2]
    r = np.linalg.norm(ecef_pts, axis=1)
    safe_r = np.where(r < 1e-6, 1.0, r)
    lat = np.degrees(np.arcsin(z / safe_r))
    lon = np.degrees(np.arctan2(y, x))
    h = r - R_E
    return lat, lon, h

def stec_via_iri(rx_itrs, sv_itrs, t_utc, ne_func):
    """integrates electron density along the line-of-sight (LOS) path."""
    dr = sv_itrs - rx_itrs
    rng = np.linalg.norm(dr)
    if rng < 1e3: return 0.0
    
    step_m = 10000.0 
    n = max(10, int(rng / step_m) + 1)
    t_steps = np.linspace(0, 1, n)
    
    # points along the LOS vector: Rx + t * (Sv - Rx)
    pts = rx_itrs[None, :] + t_steps[:, None] * dr[None, :]
    ds = rng / (n - 1)
    
    # convert to geodetic (lat/lon/h)
    lat, lon, h = ecef_to_spherical_geodetic_vec(pts)
    
    # electron density at each point
    Ne_profile = ne_func(lat, lon, h, t_utc)
    
    # trapezoidal rule for numerical integration: STEC = Integral(Ne * ds)
    return np.trapz(Ne_profile, dx=ds)

# -------------------------- MAIN -------------------------
def main(model):
    USE_IRI_LOS = (model == 'iri')
    print(f"\n[START] Running model: {model.upper()}")
    
    ts = load.timescale()
    site = wgs84.latlon(OBS_LAT_DEG, OBS_LON_DEG, elevation_m=OBS_H_M)
    t_idx, t_sky = compute_time_grid(START_UTC, END_UTC, STEP_SEC, ts)

    triples = parse_tle_text_file(HISTORICAL_TLE_FILE_CONTENT)
    if not triples: 
        print("[FATAL] Error: No valid TLE triples found in file content. Exiting.")
        sys.exit(1)
        
    sats = make_satellites_from_tles(triples, ts)

    t0 = t_sky[0]
    meo, leo_cand = [], []
    for s in sats:
        try:
            cons = constellation_name(s)
            alt0_km = satellite_altitude_m(s, t0)/1000.0
            if cons in {"GPS","Galileo","GLONASS"} and alt0_km > 10_000:
                meo.append(s)
            # filter LEO candidates by altitude and exclude known MEO/GNSS constellations
            elif alt0_km <= 2_000 and cons not in {"GPS","Galileo","GLONASS"}:
                leo_cand.append(s)
        except Exception: pass
        
    print(f"[INFO] {len(meo)} MEO satellites found")
    print(f"[INFO] {len(leo_cand)} LEO candidates (<=2000 km)")

    visible = []
    t_samples = t_sky[::SAMPLE_STRIDE]
    for sat in tqdm(leo_cand, desc="LEO screening", unit="sat"):
        try:
            topo = (sat - site).at(t_samples)
            alt = topo.altaz()[0].degrees
            max_el = np.nanmax(alt)
            if max_el >= ELEV_MASK_DEG:
                visible.append((sat, float(max_el)))
        except Exception: continue
        
    visible.sort(key=lambda x: x[1], reverse=True)
    # select the top 50 most visible LEOs
    selected_leo = [s for s,_ in visible[:50]]
    tracked = meo + selected_leo

    ne_func = hardcoded_ne_vec if USE_IRI_LOS else None
    
    # receiver ECEF position for all times
    rx_itrs_t = site.at(t_sky).frame_xyz(itrs).m.T 
    
    records = []
    for sat in tqdm(tracked, desc=f"Processing ({model})", unit="sat"):
        cons = constellation_name(sat)
        orbit_type = "MEO" if cons in {"GPS","Galileo","GLONASS"} else "LEO"
        try:
            topo = (sat - site).at(t_sky)
            alt, _, dist = topo.altaz()
            el_deg = alt.degrees
            rng_m = dist.m
            
            mask = (el_deg >= ELEV_MASK_DEG) & ~np.isnan(el_deg)
            if not np.any(mask): continue
            
            t_pd_above = t_idx[mask]
            t_sf_above = t_sky[mask]
            rx_itrs_above = rx_itrs_t[mask]
            # satellite ECEF position for the visible times
            sv_itrs_above = sat.at(t_sf_above).frame_xyz(itrs).m.T
            
            for i, (t_pd, el, rng, rx_itrs_i, sv_itrs) in enumerate(zip(t_pd_above, el_deg[mask], rng_m[mask], rx_itrs_above, sv_itrs_above)):
                if np.any(np.isnan(sv_itrs)): continue
                t_utc = t_pd.to_pydatetime()
                
                # ionospheric delay calculation
                if USE_IRI_LOS:
                    # IRI (LOS integration) model
                    stec = stec_via_iri(rx_itrs_i, sv_itrs, t_utc, ne_func)
                else:
                    # thin-shell model (standard mapping function)
                    vtec = vtec_diurnal(t_utc, vtec_max=30.0, vtec_min=5.0)
                    stec = stec_from_vtec(vtec, math.radians(el))

                # LEO altitude weighting for thin-shell model
                if orbit_type == "LEO" and not USE_IRI_LOS:
                    h_m_sat = satellite_altitude_m(sat, t_sf_above[i])
                    # factor to limit STEC contribution from low-altitude satellites
                    stec *= leo_cap_fraction(h_m_sat)

                records.append({
                    "time_utc": t_pd,
                    "satellite": sat.name,
                    "constellation": cons,
                    "orbit": orbit_type,
                    "elevation_deg": el,
                    "group_delay_m": code_group_delay_m(stec, FREQ)
                })
        except Exception as e: 
            pass

    if not records:
        print("[WARN] No records generated.")
        return

    df = pd.DataFrame.from_records(records).sort_values(["time_utc","constellation","satellite"])
    
    # -------------------------- IONO DOPPLER CALCULATION & PASS BREAKS --------------------------
    df_dop = []
    time_gap_threshold_sec = STEP_SEC * 1.5 
    
    for _, grp in df.groupby("satellite"):
        grp = grp.sort_values("time_utc").copy()
        if len(grp) > 1:
            # time difference
            dt_sec = np.diff(grp["time_utc"].astype('int64') // 10**9)
            d_delay = np.diff(grp["group_delay_m"].values)
            
        
            is_gap = dt_sec > time_gap_threshold_sec
            
            # calculate doppler: d(delay)/dt
            doppler = - (FREQ / C) * (d_delay / dt_sec)
            
            dop_vals = np.concatenate(([np.nan], doppler))
            
            for i in np.where(is_gap)[0]:
                if i + 1 < len(dop_vals):
                    dop_vals[i + 1] = np.nan 
            
            grp["iono_doppler_hz"] = dop_vals
            
            delay_vals = grp["group_delay_m"].values.copy()
            for i in np.where(is_gap)[0]:
                 if i + 1 < len(delay_vals):
                     delay_vals[i + 1] = np.nan
            grp["group_delay_m_plot"] = delay_vals
            
            df_dop.append(grp)
            
    if df_dop: 
        df = pd.concat(df_dop)
    else:
        df["group_delay_m_plot"] = df["group_delay_m"]


    # -------------------------- PLOTTING --------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 10})
    
    meo_subset = df[df['orbit']=='MEO']
    leo_subset = df[df['orbit']=='LEO']
    
    # time series plot (group delay)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plotted_meo = False
    plotted_leo = False
    
    # MEO
    if not meo_subset.empty:
        for sat_name in meo_subset['satellite'].unique(): 
            d = df[df['satellite']==sat_name]
            label = "MEO Satellites" if not plotted_meo else "_nolegend_"
            ax.plot(d['time_utc'], d['group_delay_m'], label=label, color='blue', lw=2.5, alpha=0.8) 
            plotted_meo = True
    
    # LEO 
    if not leo_subset.empty:
        for sat_name in leo_subset['satellite'].unique(): 
            d = df[df['satellite'] == sat_name]
            
            label = "LEO Satellites" if not plotted_leo else "_nolegend_"
            ax.plot(d['time_utc'], d['group_delay_m_plot'], label=label, color='red', ls='-', lw=1.5, alpha=0.6) 
            plotted_leo = True

    ax.set_ylabel("Group Delay (m)", fontweight='bold')
    ax.set_xlabel("Time (UTC)", fontweight='bold')
    ax.set_title(f"Ionospheric Group Delay: MEO vs. LEO ({model.upper()} Model) - {DATE_LABEL}", fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"plot_timeseries_{model}.png"))
    plt.close(fig)

    # heatmap: elevation vs delay
    if len(df) > 50:
        elev_bins = np.linspace(0, 90, 20)
        df['elev_bin'] = pd.cut(df['elevation_deg'], bins=elev_bins)
        
        pivot_df = df.copy() 
        pivot_df['time_utc'] = pivot_df['time_utc'].dt.floor('min')
        
        pivot = pd.pivot_table(pivot_df, values='group_delay_m', index='elev_bin', columns='time_utc', aggfunc='mean', observed=False)
        
        bin_labels = pivot.index.tolist()
        mid_elevations = []
        for cat in bin_labels:
            try:
                left, right = map(float, str(cat).strip('()[]').split(','))
                mid_elevations.append((left + right) / 2)
            except Exception:
                 mid_elevations.append(0) 

        fig_hm, ax_hm = plt.subplots(figsize=(12, 6)) 
        sns.heatmap(pivot, cmap='coolwarm', ax=ax_hm, cbar_kws={'label': 'Mean Group Delay (m)'}) 

        ax_hm.set_title(f"Heatmap: Delay (m) vs Mean Elevation ({model.upper()} Model) - {DATE_LABEL}", fontweight='bold')
        
        y_ticks_idx = np.arange(0, len(mid_elevations), 4)
        y_ticks_labels = [f"{mid_elevations[i]:.1f}°" for i in y_ticks_idx]
        
        ax_hm.set_yticks(y_ticks_idx + 0.5) 
        ax_hm.set_yticklabels(y_ticks_labels, rotation=0)

        times = pivot.columns
        step = max(1, len(times) // 10)
        ax_hm.set_xticks(range(0, len(times), step))
        ax_hm.set_xticklabels([t.strftime('%H:%M') for t in times[::step]], rotation=45, ha='right')
        
        ax_hm.set_ylabel("Mean Elevation (°)", fontweight='bold')
        ax_hm.set_xlabel("Time (UTC)", fontweight='bold')
        
        fig_hm.tight_layout()
        fig_hm.savefig(os.path.join(OUT_DIR, f"plot_heatmap_{model}.png"))
        plt.close(fig_hm)
        
    # elevation vs. delay Scatter Plot
    fig_el, ax_el = plt.subplots(figsize=(12, 6))
    
    # MEO
    if not meo_subset.empty:
        meo_scatter = meo_subset.copy()
        ax_el.plot(meo_scatter['elevation_deg'], meo_scatter['group_delay_m'], 
                  'o', label='MEO Satellites', color='blue', alpha=0.6, ms=4, markeredgecolor='none') 
        
    # LEO 
    if not leo_subset.empty:
        leo_scatter = leo_subset.copy()
        ax_el.plot(leo_scatter['elevation_deg'], leo_scatter['group_delay_m'], 
                  'o', label='LEO Satellites', color='red', alpha=0.4, ms=3, markeredgecolor='none')

    ax_el.set_xlabel("Elevation (°)", fontweight='bold')
    ax_el.set_ylabel("Group Delay (m)", fontweight='bold')
    ax_el.set_title(f"Elevation vs. Group Delay Scatter Plot ({model.upper()} Model) - {DATE_LABEL}", fontweight='bold')
    ax_el.legend(loc='lower right')
    ax_el.grid(True, linestyle='--', alpha=0.6)
    fig_el.tight_layout()
    fig_el.savefig(os.path.join(OUT_DIR, f"plot_elev_delay_{model}.png"))
    plt.close(fig_el)

    # ionospheric doppler time series 
    fig_dop, ax_dop = plt.subplots(figsize=(12, 6)) 
    plotted_meo_dop = False
    plotted_leo_dop = False
    
    # MEO
    if not meo_subset.empty:
        for sat_name in meo_subset['satellite'].unique():
            d = df[df['satellite']==sat_name]
            label = "MEO Satellites" if not plotted_meo_dop else "_nolegend_"
            ax_dop.plot(d['time_utc'], d['iono_doppler_hz'], label=label, color='blue', lw=2.5, alpha=0.8)
            plotted_meo_dop = True
    
    # LEO 
    if not leo_subset.empty:
        for sat_name in leo_subset['satellite'].unique():
            d = df[df['satellite']==sat_name]
            label = "LEO Satellites" if not plotted_leo_dop else "_nolegend_"
            ax_dop.plot(d['time_utc'], d['iono_doppler_hz'], label=label, color='red', ls='-', lw=1.5, alpha=0.6)
            plotted_leo_dop = True

    ax_dop.set_ylabel("Iono Doppler (Hz)", fontweight='bold')
    ax_dop.set_xlabel("Time (UTC)", fontweight='bold')
    ax_dop.set_title(f"Ionospheric Doppler Shift: MEO vs. LEO ({model.upper()} Model) - {DATE_LABEL}", fontweight='bold')
    ax_dop.legend(loc='upper right')
    ax_dop.grid(True, linestyle='--', alpha=0.6)
    fig_dop.tight_layout()
    fig_dop.savefig(os.path.join(OUT_DIR, f"plot_doppler_timeseries_{model}.png"))
    plt.close(fig_dop)

    
    # 5. ground tracks (cartopy)
    if tracked:
        fig_gt, ax_gt = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()}) 
        ax_gt.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_gt.add_feature(cfeature.COASTLINE)
        ax_gt.gridlines(draw_labels=True, alpha=0.3)
        ax_gt.set_extent([-180, 180, -70, 70], crs=ccrs.PlateCarree()) 

        meo_plotted, leo_plotted = False, False
        
        # MEO
        for sat in meo:
            sub = sat.at(t_sky).subpoint()
            lat = sub.latitude.degrees
            lon = sub.longitude.degrees
            
            diff = np.diff(lon)
            jumps = np.where(np.abs(diff) > 180)[0] + 1
            segments = np.split(np.arange(len(lon)), jumps)
            
            label_text = "MEO Tracks (Blue)" if not meo_plotted else "_nolegend_"
            
            for seg in segments:
                if len(seg) > 0:
                    ax_gt.plot(lon[seg], lat[seg], transform=ccrs.PlateCarree(), 
                               color='blue', lw=2.5, ls='-', alpha=0.8, label=label_text if seg[0] == segments[0][0] else "_nolegend_")
            meo_plotted = True

        # LEO
        for sat in selected_leo[:10]:
            sub = sat.at(t_sky).subpoint()
            lat = sub.latitude.degrees
            lon = sub.longitude.degrees
            
            diff = np.diff(lon)
            jumps = np.where(np.abs(diff) > 180)[0] + 1
            segments = np.split(np.arange(len(lon)), jumps)
            
            label_text = "LEO Tracks (Red, Dashed)" if not leo_plotted else "_nolegend_"

            for seg in segments:
                if len(seg) > 0:
                    ax_gt.plot(lon[seg], lat[seg], transform=ccrs.PlateCarree(), 
                               color='red', lw=1.5, ls='--', alpha=0.6, label=label_text if seg[0] == segments[0][0] else "_nolegend_")
            leo_plotted = True

        handles, labels = ax_gt.get_legend_handles_labels()
        unique_labels = {}
        for h, l in zip(handles, labels):
            if l not in unique_labels and l != "_nolegend_":
                unique_labels[l] = h

        if unique_labels:
            ax_gt.legend(unique_labels.values(), unique_labels.keys(), loc='lower left', fontsize='small')
        
        ax_gt.set_title(f"Satellite Ground Tracks - {DATE_LABEL}", fontweight='bold')
        fig_gt.tight_layout()
        fig_gt.savefig(os.path.join(OUT_DIR, f"plot_groundtracks_{model}.png"))
        plt.close(fig_gt)
        
    # total number of plots created: timeseries (1), heatmap (1), ele vs delay (1), doppler (1), ground tracks (1) = 5 per model
    print(f"Saved 5 plots to {OUT_DIR} for model {model.upper()}.")

if __name__ == "__main__":
    for m in ['thin', 'iri']:
        main(m)
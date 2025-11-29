import numpy as np
import matplotlib.pyplot as plt

# -------------------------- CONFIG --------------------------
FONT_SIZE = 16
LINE_WIDTH = 3
MARKER_SIZE = 10 
plt.rcParams.update({'font.size': FONT_SIZE}) 

# altitude data (h in km)
H_min = 40
H_max = 500
altitude = np.linspace(H_min, H_max, 500)

# gaussian-like layer profile
def layer_log_density(h, peak_h, peak_log_Ne, width_h):
    return peak_log_Ne * np.exp(-((h - peak_h) / width_h)**2)

# -------------------------- IONO LAYER PARAMETERS -------------------------- 
# (Log10(Ne) in el/m^3)
# D Layer: ~75 km
# E Layer: ~110 km
# F1 Layer: ~180 km (day only)
# F2 Layer: ~300-350 km (main peak)

# -------------------------- DAYTIME PROFILE --------------------------
# D Layer: prominent in day (Lyman-alpha)
day_D_log_Ne = layer_log_density(altitude, 75, 9.8, 15)
# E Layer: strong in day (soft X-ray/EUV)
day_E_log_Ne = layer_log_density(altitude, 110, 10.6, 25)
# F1 Layer: distinct ledge in day (EUV)
day_F1_log_Ne = layer_log_density(altitude, 180, 11.2, 40)
# F2 Layer: main, highest peak in day (EUV)
day_F2_log_Ne = layer_log_density(altitude, 300, 12.0, 70)

base_log_density_day = 9.0 * np.exp(-altitude / 200) # base density decreasing with altitude
log_Ne_day = day_D_log_Ne + day_E_log_Ne + day_F1_log_Ne + day_F2_log_Ne + base_log_density_day

# -------------------------- NIGHTTIME PROFILE --------------------------
# D Layer: virtually disappears at night
night_D_log_Ne = layer_log_density(altitude, 75, 8.0, 15)
# E Layer: weakens significantly, remains as a trace
night_E_log_Ne = layer_log_density(altitude, 110, 9.0, 30)
# F1 Layer: merges with F2 layer
night_F1_log_Ne = layer_log_density(altitude, 180, 0.0, 1)
# F2 Layer: single F layer remains (higher altitude, lower peak density than day F2)
night_F2_log_Ne = layer_log_density(altitude, 350, 11.5, 90)

base_log_density_night = 8.0 * np.exp(-altitude / 300) # lower base density
log_Ne_night = night_D_log_Ne + night_E_log_Ne + night_F2_log_Ne + base_log_density_night

# -------------------------- PLOTTING --------------------------
plt.figure(figsize=(10, 7))

# daytime plot
plt.plot(10**log_Ne_day, altitude, label='Daytime Profile', color='red', linewidth=2)
# nighttime plot
plt.plot(10**log_Ne_night, altitude, label='Nighttime Profile', color='blue', linestyle='--', linewidth=2)

plt.xscale('log')
plt.xlabel(r'Plasma Density $N_e$ ($\mathrm{e}^-/\mathrm{m}^3$)', fontsize=FONT_SIZE + 4)
plt.ylabel('Altitude (km)', fontsize=FONT_SIZE + 4)
plt.title('Ionospheric Stratification', fontsize=FONT_SIZE + 8, fontweight='bold')

plt.legend(loc='upper right', fontsize=FONT_SIZE, framealpha=0.8)
plt.grid(True, which="both", ls="--", linewidth=1) 

plt.ylim(H_min, H_max)
plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
plt.tight_layout()
plt.savefig('ionosphere_profile_presentation.png', dpi=300)

print("ionosphere_profile_presentation.png")
plt.show()
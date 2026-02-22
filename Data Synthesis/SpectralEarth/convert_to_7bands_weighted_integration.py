import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm

# ============================================================
# 1) SpectralEarth / EnMAP wavelength grid (nm)
#    MUST match the channel order in the TIFFs
# ============================================================

wavelengths_spectralearth = np.array([
    418.42, 424.04, 429.46, 434.69, 439.76, 444.70, 449.54, 454.31, 459.03, 463.73,
    468.41, 473.08, 477.74, 482.41, 487.09, 491.78, 496.50, 501.24, 506.02, 510.83,
    515.67, 520.55, 525.47, 530.42, 535.42, 540.46, 545.55, 550.69, 555.87, 561.11,
    566.41, 571.76, 577.17, 582.64, 588.17, 593.77, 599.45, 605.19, 611.02, 616.92,
    622.92, 628.99, 635.11, 641.29, 647.54, 653.84, 660.21, 666.64, 673.13, 679.69,
    686.32, 693.01, 699.78, 706.62, 713.52, 720.50, 727.55, 734.65, 741.83, 749.06,
    756.35, 763.70, 771.11, 778.57, 786.08, 793.64, 801.25, 808.91, 816.61, 824.36,
    832.15, 839.98, 847.85, 855.76, 863.70, 871.68, 879.69, 887.73, 895.79,
    901.96, 903.87, 911.57, 911.97, 920.08, 921.32, 928.20, 931.20, 936.34,
    941.22, 944.47, 951.36, 952.61, 960.75, 961.63, 968.89, 972.02, 977.04,
    982.52, 985.19, 993.14, 993.34, 1003.88, 1014.72, 1025.66, 1036.70,
    1047.84, 1059.07, 1070.39, 1081.78, 1093.26, 1104.81, 1116.43, 1128.10,
    1139.84, 1151.62, 1163.44, 1175.30, 1187.20, 1199.11, 1211.05, 1223.00,
    1234.97, 1246.94, 1258.93, 1270.92, 1282.92,
    1530.29, 1541.67, 1553.01, 1564.30, 1575.55, 1586.76, 1597.91, 1609.02,
    1620.09, 1631.11, 1642.07, 1653.00, 1663.87, 1674.70, 1685.47, 1696.20,
    1706.87, 1717.50, 1728.08,
    1977.08, 1986.45, 1995.79, 2005.08, 2014.33, 2023.54, 2032.70, 2041.83,
    2050.92, 2059.96, 2068.97, 2077.93, 2086.86, 2095.74, 2104.59, 2113.40,
    2122.17, 2130.90, 2139.60, 2148.26, 2156.88, 2165.47, 2174.02, 2182.53,
    2191.01, 2199.45, 2207.86, 2216.24, 2224.58, 2232.89, 2241.16, 2249.40,
    2257.61, 2265.79, 2273.93, 2282.04, 2290.12, 2298.17, 2306.19, 2314.17,
    2322.13, 2330.05, 2337.94, 2345.81, 2353.64, 2361.44, 2369.21, 2376.95,
    2384.66, 2392.34, 2400.00, 2407.62, 2415.21, 2422.78, 2430.32, 2437.82,
    2445.30
], dtype=np.float32)

# spectral step Δλ for integration
DL_ENMAP = np.gradient(wavelengths_spectralearth).astype(np.float32)

# ============================================================
# 2) Load SRF CSV (wavelength_nm, srf)
# ============================================================

def load_srf_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float32)
    wl = data[:, 0]
    srf = data[:, 1]

    order = np.argsort(wl)
    wl = wl[order]
    srf = srf[order]

    return wl, np.clip(srf, 0.0, None)


def interpolate_srf_to_enmap(lambda_srf, srf, lambda_enmap):
    return np.interp(lambda_enmap, lambda_srf, srf, left=0.0, right=0.0).astype(np.float32)


def integrate_band(img_hwc, srf_enmap):
    """
    SRF-weighted spectral integral
    """
    num = np.sum(
        img_hwc * (srf_enmap[None, None, :] * DL_ENMAP[None, None, :]),
        axis=2
    )
    den = np.sum(srf_enmap * DL_ENMAP)

    if den <= 0:
        raise ValueError("SRF integral is zero (check SRF overlap).")

    return (num / den).astype(np.float32)

# ============================================================
# 3) Landsat-8 SRF files
# ============================================================

L8_SRF_DIR = Path("L8_SRF")

L8_SRF_FILES = [
    L8_SRF_DIR / "L8_B1.csv",
    L8_SRF_DIR / "L8_B2.csv",
    L8_SRF_DIR / "L8_B3.csv",
    L8_SRF_DIR / "L8_B4.csv",
    L8_SRF_DIR / "L8_B5.csv",
    L8_SRF_DIR / "L8_B6.csv",
    L8_SRF_DIR / "L8_B7.csv",
]


def load_all_srfs():
    srfs = []
    for path in L8_SRF_FILES:
        wl, srf = load_srf_csv(path)
        srf_enmap = interpolate_srf_to_enmap(wl, srf, wavelengths_spectralearth)

        if np.sum(srf_enmap) == 0:
            raise RuntimeError(f"No overlap for SRF {path.name}")

        srfs.append(srf_enmap)

    return np.stack(srfs, axis=0)

# ============================================================
# 4) Conversion
# ============================================================

def convert_spectralearth_to_landsat8_srf(input_folder: str):
    input_path = Path(input_folder)
    output_path = input_path.parent / f"{input_path.name}_7bands_srf"
    output_path.mkdir(parents=True, exist_ok=True)

    srfs = load_all_srfs()
    print(f"[INFO] Loaded {srfs.shape[0]} Landsat-8 SRFs")

    for folder in sorted(input_path.iterdir()):
        if not folder.is_dir():
            continue

        out_sub = output_path / folder.name
        out_sub.mkdir(parents=True, exist_ok=True)

        for tif in tqdm(sorted(folder.glob("*.tif")), desc=folder.name):
            img = tifffile.imread(tif).astype(np.float32)

            if img.ndim != 3 or img.shape[2] != len(wavelengths_spectralearth):
                raise ValueError(f"Bad shape {img.shape} in {tif.name}")

            bands = [integrate_band(img, srfs[i]) for i in range(srfs.shape[0])]
            out = np.stack(bands, axis=2)

            tifffile.imwrite(out_sub / tif.name, out)

    print(f"\n✅ Conversion completed: {output_path}")

# ============================================================
# 5) Run
# ============================================================

if __name__ == "__main__":
    convert_spectralearth_to_landsat8_srf("enmap_subset")
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm

# SpectralEarth wavelengths
wavelengths_spectralearth = np.array([
    418.42, 424.04, 429.46, 434.69, 439.76, 444.70, 449.54, 454.31, 459.03, 463.73, 468.41, 473.08,
    477.74, 482.41, 487.09, 491.78, 496.5, 501.24, 506.02, 510.83, 515.67, 520.55, 525.47, 530.42,
    535.42, 540.46, 545.55, 550.69, 555.87, 561.11, 566.41, 571.76, 577.17, 582.64, 588.17, 593.77,
    599.45, 605.19, 611.02, 616.92, 622.92, 628.99, 635.11, 641.29, 647.54, 653.84, 660.21, 666.64,
    673.13, 679.69, 686.32, 693.01, 699.78, 706.62, 713.52, 720.50, 727.55, 734.65, 741.83, 749.06,
    756.35, 763.70, 771.11, 778.57, 786.08, 793.64, 801.25, 808.91, 816.61, 824.36, 832.15, 839.98,
    847.85, 855.76, 863.70, 871.68, 879.69, 887.73, 895.79, 901.96, 903.87, 911.57, 911.97, 920.08,
    921.32, 928.20, 931.20, 936.34, 941.22, 944.47, 951.36, 952.61, 960.75, 961.63, 968.89, 972.02,
    977.04, 982.52, 985.19, 993.14, 993.34, 1003.88, 1014.72, 1025.66, 1036.70, 1047.84, 1059.07,
    1070.39, 1081.78, 1093.26, 1104.81, 1116.43, 1128.10, 1139.84, 1151.62, 1163.44, 1175.30,
    1187.20, 1199.11, 1211.05, 1223.00, 1234.97, 1246.94, 1258.93, 1270.92, 1282.92, 1530.29,
    1541.67, 1553.01, 1564.30, 1575.55, 1586.76, 1597.91, 1609.02, 1620.09, 1631.11, 1642.07,
    1653.00, 1663.87, 1674.70, 1685.47, 1696.20, 1706.87, 1717.50, 1728.08, 1977.08, 1986.45,
    1995.79, 2005.08, 2014.33, 2023.54, 2032.70, 2041.83, 2050.92, 2059.96, 2068.97, 2077.93,
    2086.86, 2095.74, 2104.59, 2113.40, 2122.17, 2130.90, 2139.60, 2148.26, 2156.88, 2165.47,
    2174.02, 2182.53, 2191.01, 2199.45, 2207.86, 2216.24, 2224.58, 2232.89, 2241.16, 2249.40,
    2257.61, 2265.79, 2273.93, 2282.04, 2290.12, 2298.17, 2306.19, 2314.17, 2322.13, 2330.05,
    2337.94, 2345.81, 2353.64, 2361.44, 2369.21, 2376.95, 2384.66, 2392.34, 2400.00, 2407.62,
    2415.21, 2422.78, 2430.32, 2437.82, 2445.30
], dtype=np.float32)

# Landsat 8 band wavelength ranges
landsat_band_ranges = [
    (435, 451),   # Band 1
    (452, 512),   # Band 2
    (533, 590),   # Band 3
    (646, 673),   # Band 4
    (851, 879),   # Band 5
    (1556, 1651), # Band 6
    (2107, 2294), # Band 7
]

def get_band_indices(hyper_wl, landsat_ranges):
    bands = []
    for lower, upper in landsat_ranges:
        indices = np.where((hyper_wl >= lower) & (hyper_wl <= upper))[0]
        if len(indices) == 0:
            raise ValueError(f"No bands found in range {lower}-{upper}nm.")
        bands.append(indices)
    return bands

band_indices = get_band_indices(wavelengths_spectralearth, landsat_band_ranges)

def convert_spectralearth_to_landsat8(input_folder: str):
    input_path = Path(input_folder)
    output_path = input_path.parent / f"{input_path.name}_7bands"
    skipped_files = []

    for folder in sorted(input_path.iterdir()):
        if not folder.is_dir():
            continue

        out_subfolder = output_path / folder.name
        out_subfolder.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(folder.glob("*.tif"))
        for tif_file in tqdm(tif_files, desc=f"Processing {folder.name}"):
            try:
                img = tifffile.imread(str(tif_file))  # (H, W, C)

                # Skip images with unexpected dimensions
                if img.ndim != 3 or img.shape[2] < max([i.max() for i in band_indices]) + 1:
                    raise ValueError(f"Invalid shape: {img.shape}")

                out_bands = []
                for band_group in band_indices:
                    band_avg = img[:, :, band_group].mean(axis=2)
                    out_bands.append(band_avg)

                img_7bands = np.stack(out_bands, axis=2)  # (H, W, 7)

                out_file = out_subfolder / tif_file.name
                tifffile.imwrite(out_file, img_7bands.astype(np.float32))

            except Exception as e:
                print(f"⚠️ Skipping {tif_file} due to error: {e}")
                skipped_files.append(str(tif_file))

    print(f"\n✅ All valid images converted and saved to: {output_path}")
    if skipped_files:
        print(f"\n⚠️ Skipped {len(skipped_files)} corrupted or invalid files.")
        for f in skipped_files:
            print(f" - {f}")

# Example usage:
convert_spectralearth_to_landsat8("enmap_subset")

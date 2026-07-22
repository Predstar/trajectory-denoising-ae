"""Inspect saved clean/noisy/reconstructed trajectory datasets.

Run: python inspect_data.py [sample_idx] [--csv]
  sample_idx  which trajectory to preview/export (default 0)
  --csv       also export that sample to data/sample_<idx>.csv
"""

import sys
import os
import csv
import numpy as np

DATA_DIR = "data"


def load_datasets(data_dir=DATA_DIR):
    clean = np.load(os.path.join(data_dir, "clean_trajectories.npy"))
    noisy = np.load(os.path.join(data_dir, "noisy_trajectories.npy"))
    recon_path = os.path.join(data_dir, "reconstructed_trajectories.npy")
    recon = np.load(recon_path) if os.path.exists(recon_path) else None
    return clean, noisy, recon


def summarize(name, arr):
    print(f"{name}: shape={arr.shape}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")


def export_sample_csv(clean, noisy, recon, sample_idx, data_dir=DATA_DIR):
    """Write one trajectory's clean/noisy/reconstructed x,y,z to a single CSV."""
    T = clean.shape[1]
    has_recon = recon is not None and sample_idx < len(recon)

    header = ["t", "clean_x", "clean_y", "clean_z", "noisy_x", "noisy_y", "noisy_z"]
    if has_recon:
        header += ["recon_x", "recon_y", "recon_z"]

    out_path = os.path.join(data_dir, f"sample_{sample_idx}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in range(T):
            row = [t, *clean[sample_idx, t], *noisy[sample_idx, t]]
            if has_recon:
                row += list(recon[sample_idx, t])
            writer.writerow(row)

    print(f"Saved {out_path}")


def main():
    args = sys.argv[1:]
    export_csv = "--csv" in args
    args = [a for a in args if a != "--csv"]
    sample_idx = int(args[0]) if args else 0

    clean, noisy, recon = load_datasets()

    print("=== Dataset summary ===")
    summarize("clean", clean)
    summarize("noisy", noisy)
    if recon is not None:
        summarize("reconstructed", recon)
    else:
        print("reconstructed: not found (run main.py first)")

    print(f"\n=== Sample {sample_idx}, first 5 timesteps (x, y, z) ===")
    print("clean:\n", clean[sample_idx, :5])
    print("noisy:\n", noisy[sample_idx, :5])
    if recon is not None and sample_idx < len(recon):
        print("reconstructed:\n", recon[sample_idx, :5])

    if export_csv:
        export_sample_csv(clean, noisy, recon, sample_idx)


if __name__ == "__main__":
    main()
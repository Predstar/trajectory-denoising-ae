

"""Animate a single trajectory (clean vs noisy vs reconstructed) as an MP4.

Not part of AE training itself -- just a visual sanity check. Loads a sample
from the saved dataset (data/clean_trajectories.npy, data/noisy_trajectories.npy,
data/reconstructed_trajectories.npy) if available; otherwise generates a fresh
clean/noisy trajectory on the fly (no reconstructed line in that fallback case,
since there's no trained model to produce one).

MP4 is used instead of GIF because most OS-default viewers (e.g. Windows
Photos) only display a GIF's first frame instead of auto-playing it, while
video players auto-play MP4 reliably. Uses the ffmpeg binary bundled by the
imageio-ffmpeg package, so no system-wide ffmpeg install is required.

Run standalone: python animate_preview.py
Or import animate_sample() and call it from another script (e.g. main.py).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation, FFMpegWriter

import imageio_ffmpeg
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

from trajectory import generate_spiral_trajectory
from noise import add_gaussian_noise

DATA_DIR = "data"
SAMPLE_IDX = 0  # which trajectory to animate when loading from the saved dataset


def animate_sample(data_dir=DATA_DIR, sample_idx=SAMPLE_IDX, out_path="trajectory_anim.mp4",
                    max_frames=100, recon_idx=None):
    """Load (or generate) a clean/noisy/reconstructed trajectory triple and
    save it as an animated GIF.

    Args:
        sample_idx: index into the full clean/noisy dataset.
        recon_idx: index into reconstructed_trajectories.npy, which is only
            saved for the *test* split and is therefore indexed separately
            from sample_idx (e.g. recon_idx = sample_idx - n_train). If None,
            no reconstructed line is drawn.
        max_frames: trajectory is subsampled down to at most this many frames,
            so render time stays fast regardless of how large T is.
    """
    clean_path = os.path.join(data_dir, "clean_trajectories.npy")
    noisy_path = os.path.join(data_dir, "noisy_trajectories.npy")
    recon_path = os.path.join(data_dir, "reconstructed_trajectories.npy")

    recon = None
    if os.path.exists(clean_path) and os.path.exists(noisy_path):
        print(f"Loading sample {sample_idx} from saved dataset in {data_dir}/")
        traj = np.load(clean_path)[sample_idx]
        noisy = np.load(noisy_path)[sample_idx]
        if recon_idx is not None and os.path.exists(recon_path):
            recon = np.load(recon_path)[recon_idx]
    else:
        print("No saved dataset found, generating a fresh trajectory")
        traj = generate_spiral_trajectory(T=200)
        noisy = add_gaussian_noise(traj, sigma=0.15)

    if len(traj) > max_frames:
        idx = np.linspace(0, len(traj) - 1, max_frames).astype(int)
        traj = traj[idx]
        noisy = noisy[idx]
        if recon is not None:
            recon = recon[idx]

    n = len(traj)
    # Sequential phases: clean draws fully, then noisy draws fully (clean stays
    # visible), then reconstructed draws fully (clean+noisy stay visible).
    # Each phase gets its own frame range so they play one after another.
    phases = [("clean", n)]
    phases.append(("noisy", n))
    if recon is not None:
        phases.append(("recon", n))
    phase_bounds = np.cumsum([0] + [count for _, count in phases])

    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_subplot(projection='3d')
    ax.set_facecolor("black")
    ax.xaxis.set_pane_color((0, 0, 0, 1))
    ax.yaxis.set_pane_color((0, 0, 0, 1))
    ax.zaxis.set_pane_color((0, 0, 0, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color("white")
        axis.label.set_color("white")
    ax.tick_params(colors="white")
    ax.view_init(elev=22, azim=-60)

    all_points = [traj, noisy] if recon is None else [traj, noisy, recon]
    all_points = np.concatenate(all_points, axis=0)
    ax.set_xlim(all_points[:, 0].min() - 0.5, all_points[:, 0].max() + 0.5)
    ax.set_ylim(all_points[:, 1].min() - 0.5, all_points[:, 1].max() + 0.5)
    ax.set_zlim(all_points[:, 2].min() - 0.5, all_points[:, 2].max() + 0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    clean_line, = ax.plot([], [], [], lw=2, color="#3987e5", label='clean')
    clean_point, = ax.plot([], [], [], 'o', color="#3987e5")
    noisy_line, = ax.plot([], [], [], lw=1.3, alpha=0.8, color="#d95926", label='noisy')
    noisy_point, = ax.plot([], [], [], 'o', color="#d95926")
    if recon is not None:
        recon_line, = ax.plot([], [], [], lw=2, linestyle='--', color="#199e70", label='reconstructed')
        recon_point, = ax.plot([], [], [], 'o', color="#199e70")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

    def update(frame):
        # Which phase are we in, and how many points of that phase's data to draw.
        phase_idx = np.searchsorted(phase_bounds, frame, side='right') - 1
        phase_idx = min(phase_idx, len(phases) - 1)
        local_frame = frame - phase_bounds[phase_idx]

        clean_upto = n if phase_idx > 0 else local_frame
        clean_line.set_data(traj[:clean_upto, 0], traj[:clean_upto, 1])
        clean_line.set_3d_properties(traj[:clean_upto, 2])
        if clean_upto > 0:
            clean_point.set_data(traj[clean_upto - 1:clean_upto, 0], traj[clean_upto - 1:clean_upto, 1])
            clean_point.set_3d_properties(traj[clean_upto - 1:clean_upto, 2])

        noisy_upto = 0 if phase_idx < 1 else (n if phase_idx > 1 else local_frame)
        noisy_line.set_data(noisy[:noisy_upto, 0], noisy[:noisy_upto, 1])
        noisy_line.set_3d_properties(noisy[:noisy_upto, 2])
        if noisy_upto > 0:
            noisy_point.set_data(noisy[noisy_upto - 1:noisy_upto, 0], noisy[noisy_upto - 1:noisy_upto, 1])
            noisy_point.set_3d_properties(noisy[noisy_upto - 1:noisy_upto, 2])

        if recon is None:
            return clean_line, clean_point, noisy_line, noisy_point

        recon_upto = 0 if phase_idx < 2 else local_frame
        recon_line.set_data(recon[:recon_upto, 0], recon[:recon_upto, 1])
        recon_line.set_3d_properties(recon[:recon_upto, 2])
        if recon_upto > 0:
            recon_point.set_data(recon[recon_upto - 1:recon_upto, 0], recon[recon_upto - 1:recon_upto, 1])
            recon_point.set_3d_properties(recon[recon_upto - 1:recon_upto, 2])

        return clean_line, clean_point, noisy_line, noisy_point, recon_line, recon_point

    total_frames = phase_bounds[-1]
    anim = FuncAnimation(fig, update, frames=range(1, total_frames + 1), interval=30)
    anim.save(out_path, writer=FFMpegWriter(fps=15), savefig_kwargs={"facecolor": "black"})
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    animate_sample()
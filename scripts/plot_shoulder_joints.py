"""
Animate shoulder joint angles from a bmimic NPZ file at 50 fps.
Sliding-window mode: x-axis scrolls, window width = 1.5 s.

Left shoulder  → blue  family (dark/mid/light for pitch/roll/yaw)
Right shoulder → red   family (dark/mid/light for pitch/roll/yaw)

Usage:
    python scripts/plot_shoulder_joints.py [--npz_file PATH] [--output PATH]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Joint index mapping (lab_joint_name_list order, 0-based)
# ---------------------------------------------------------------------------
lab_joint_name_list = [
    "left_hip_pitch_joint",      # 0
    "right_hip_pitch_joint",     # 1
    "waist_yaw_joint",           # 2
    "left_hip_roll_joint",       # 3
    "right_hip_roll_joint",      # 4
    "waist_roll_joint",          # 5
    "left_hip_yaw_joint",        # 6
    "right_hip_yaw_joint",       # 7
    "waist_pitch_joint",         # 8
    "left_knee_joint",           # 9
    "right_knee_joint",          # 10
    "left_shoulder_pitch_joint", # 11
    "right_shoulder_pitch_joint",# 12
    "left_ankle_pitch_joint",    # 13
    "right_ankle_pitch_joint",   # 14
    "left_shoulder_roll_joint",  # 15
    "right_shoulder_roll_joint", # 16
    "left_ankle_roll_joint",     # 17
    "right_ankle_roll_joint",    # 18
    "left_shoulder_yaw_joint",   # 19
    "right_shoulder_yaw_joint",  # 20
    "left_elbow_joint",          # 21
    "right_elbow_joint",         # 22
    "left_wrist_roll_joint",     # 23
    "right_wrist_roll_joint",    # 24
    "left_wrist_pitch_joint",    # 25
    "right_wrist_pitch_joint",   # 26
    "left_wrist_yaw_joint",      # 27
    "right_wrist_yaw_joint",     # 28
]

IDX = {name: i for i, name in enumerate(lab_joint_name_list)}

LEFT_JOINTS  = ["left_shoulder_pitch_joint",  "left_shoulder_roll_joint",  "left_shoulder_yaw_joint"]
RIGHT_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint"]

# Joint limits (rad) — kept for ylim computation only, not drawn
LIMITS = {
    "left_shoulder_pitch_joint":  (-3.0892,  2.6704),
    "right_shoulder_pitch_joint": (-3.0892,  2.6704),
    "left_shoulder_roll_joint":   (-1.5882,  2.2515),
    "right_shoulder_roll_joint":  (-2.2515,  1.5882),
    "left_shoulder_yaw_joint":    (-2.618,   2.618),
    "right_shoulder_yaw_joint":   (-2.618,   2.618),
}

# Color palettes (dark → mid → light for pitch → roll → yaw)
LEFT_COLORS  = ["#1f4e79", "#2e75b6", "#9dc3e6"]   # blue family
RIGHT_COLORS = ["#7f0000", "#c00000", "#ff9999"]   # red family

SHORT_NAMES = ["pitch", "roll", "yaw"]

WINDOW = 1.5   # sliding window width in seconds

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DEFAULT_NPZ = (
    "/home/amax/devel/dataset/NeoBot/"
    "momillion_selected_for_web_gmr/joint_jump/gmr/002778.npz"
)

parser = argparse.ArgumentParser(description="Animate shoulder joint angles.")
parser.add_argument("--npz_file", default=DEFAULT_NPZ)
parser.add_argument("--output",   default="shoulder_joints.mp4")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data      = np.load(args.npz_file, allow_pickle=True)
joint_pos = data["joint_pos"]          # (N, 29)
fps       = float(data["fps"])
N         = len(joint_pos)
time      = np.arange(N) / fps

left_angles  = np.stack([joint_pos[:, IDX[j]] for j in LEFT_JOINTS],  axis=1)  # (N,3)
right_angles = np.stack([joint_pos[:, IDX[j]] for j in RIGHT_JOINTS], axis=1)  # (N,3)

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
plt.rc("font", family="DejaVu Sans")
fig, (ax_l, ax_r) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
# fig.suptitle("Shoulder Joint Angles", fontsize=24, fontweight="bold")

def compute_ylim(joints, angles):
    all_lo  = min(LIMITS[j][0] for j in joints)
    all_hi  = max(LIMITS[j][1] for j in joints)
    data_lo = angles.min()
    data_hi = angles.max()
    margin  = 0.15
    return min(all_lo, data_lo) - margin, max(all_hi, data_hi) + margin

ylim_l = compute_ylim(LEFT_JOINTS,  left_angles)
ylim_r = compute_ylim(RIGHT_JOINTS, right_angles)

for ax, title, ylim in [(ax_l, "Left Shoulder", ylim_l),
                        (ax_r, "Right Shoulder", ylim_r)]:
    ax.set_title(title, fontsize=20)
    ax.set_ylabel("Angle (rad)", fontsize=18)
    ax.tick_params(axis="both", labelsize=17)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, WINDOW)
    ax.set_ylim(*ylim)

ax_r.set_xlabel("Time (s)", fontsize=18)

# ---------------------------------------------------------------------------
# Animated line objects and cursor lines
# ---------------------------------------------------------------------------
LW = 4.0   # line width

left_lines  = [ax_l.plot([], [], color=c, linewidth=LW, label=f"left_{n}")[0]
               for c, n in zip(LEFT_COLORS, SHORT_NAMES)]
right_lines = [ax_r.plot([], [], color=c, linewidth=LW, label=f"right_{n}")[0]
               for c, n in zip(RIGHT_COLORS, SHORT_NAMES)]

# Cursor at the right edge of the window (current time)
cursor_l = ax_l.axvline(WINDOW, color="gray", linewidth=1.2, linestyle=":")
cursor_r = ax_r.axvline(WINDOW, color="gray", linewidth=1.2, linestyle=":")

ax_l.legend(loc="upper left", fontsize=17, framealpha=0.7)
ax_r.legend(loc="upper left", fontsize=17, framealpha=0.7)

fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.09, hspace=0.30)

# ---------------------------------------------------------------------------
# Animation  (sliding window: x-axis = [t_now - WINDOW, t_now])
# ---------------------------------------------------------------------------
STEP = max(1, int(fps / 50))   # subsample to nominal 50 fps output

def init():
    for ln in left_lines + right_lines:
        ln.set_data([], [])
    return left_lines + right_lines + [cursor_l, cursor_r]


def update(frame_idx):
    t_now  = time[frame_idx]
    t_lo   = max(0.0, t_now - WINDOW)
    t_hi   = t_lo + WINDOW

    # Mask for frames inside the window
    mask = (time >= t_lo) & (time <= t_hi)
    t_win = time[mask]

    for k, ln in enumerate(left_lines):
        ln.set_data(t_win, left_angles[mask, k])
    for k, ln in enumerate(right_lines):
        ln.set_data(t_win, right_angles[mask, k])

    # Scroll the x-axis
    ax_l.set_xlim(t_lo, t_hi)
    ax_r.set_xlim(t_lo, t_hi)

    # Cursor at the leading edge
    cursor_l.set_xdata([t_now])
    cursor_r.set_xdata([t_now])

    return left_lines + right_lines + [cursor_l, cursor_r]


frames = range(0, N, STEP)
ani = animation.FuncAnimation(
    fig, update, frames=frames,
    init_func=init, blit=False,   # blit=False because xlim changes
    interval=1000.0 / fps,
)

print(f"Saving animation to {args.output}  ({N} frames @ {fps:.0f} fps) ...")
ani.save(
    args.output,
    writer=animation.FFMpegWriter(fps=fps, bitrate=2000),
    dpi=150,
)
print("Done.")
plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt

lab_joint_name_list = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

WAIST_ROLL_IDX = lab_joint_name_list.index("right_shoulder_roll_joint")  # 15

# Joint limits from assets/unitree_g1/g1_custom_collision_29dof.urdf
JOINT_LIMIT_LOWER = -2.25  # rad
JOINT_LIMIT_UPPER =  1.58  # rad

files = [
    ("NMR",      "/home/amax/devel/dataset/NeoBot/neobot-testset/from_neobot_2/ulom/long/D2_-_Wait_1_stageii.npz"),
    ("PHUMA","/home/amax/devel/dataset/NeoBot/neobot-testset/from_phuma_bmimic/D2_-_Wait_1_stageii.npz"),
    ("GMR",  "/home/amax/devel/dataset/NeoBot/neobot-testset/from_gmr_bmimic/ulom/long/D2_-_Wait_1_stageii.npz"),
]

plt.rc('font', family='Times New Roman')
fig, ax = plt.subplots(figsize=(8, 3))

# T_START = 21.24
# T_DURATION = 1.4

T_START = 28.04
T_DURATION = 1
for label, path in files:
    d = np.load(path)
    fps = float(d["fps"])
    angles = d["joint_pos"][:, WAIST_ROLL_IDX]
    t = np.arange(len(angles)) / fps - T_START
    ax.plot(t, angles, label=label, linewidth=4)

# ax.axhline(JOINT_LIMIT_UPPER, color="red", linestyle="--", linewidth=1.5,
#            label=f"upper limit ({JOINT_LIMIT_UPPER} rad)")
ax.axhline(JOINT_LIMIT_LOWER, color="red", linestyle=":",  linewidth=4,
           label=f"lower limit ({JOINT_LIMIT_LOWER} rad)")

ax.set_xlabel("Time (s)", fontsize=15)
ax.set_ylabel("R-Shoulder Roll (rad)", fontsize=15)
ax.tick_params(axis='both', labelsize=15)
ax.leg = ax.legend(ncol=5, loc='lower center', bbox_to_anchor=(0.5, 0.95),
                   fontsize=15, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, T_DURATION])
# ax.set_ylim([-0.10, 0.55])
fig.subplots_adjust(left=0.090, right=0.980, top=0.865, bottom=0.183)

save_path = "waist_roll_comparison.pdf"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved to {save_path}")
plt.show()

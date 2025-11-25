# SMPL-X 动作可视化工具使用说明

## 概述
现在你可以用 matplotlib 火柴人来可视化 SMPL-X 动作数据了！

## 安装依赖
```bash
pip install matplotlib
```

## 使用方法

### 方法1：在 smplx_to_robot.py 中直接可视化

```bash
# 交互式模式（推荐）- 使用滑块浏览每一帧
python scripts/smplx_to_robot.py \
    --smplx_file /path/to/your/motion.npz \
    --robot unitree_g1 \
    --visualize_smplx \
    --viz_mode interactive

# 动画模式 - 自动播放动画
python scripts/smplx_to_robot.py \
    --smplx_file /path/to/your/motion.npz \
    --robot unitree_g1 \
    --visualize_smplx \
    --viz_mode animation

# 网格模式 - 显示6个关键帧的静态图
python scripts/smplx_to_robot.py \
    --smplx_file /path/to/your/motion.npz \
    --robot unitree_g1 \
    --visualize_smplx \
    --viz_mode grid
```

### 方法2：使用独立的可视化脚本

```bash
# 交互式可视化（可以用滑块控制帧）
python scripts/visualize_smplx_stickfigure.py \
    --smplx_file /path/to/your/motion.npz \
    --mode 3d

# 创建动画并播放
python scripts/visualize_smplx_stickfigure.py \
    --smplx_file /path/to/your/motion.npz \
    --mode 3d \
    --animate

# 保存为GIF
python scripts/visualize_smplx_stickfigure.py \
    --smplx_file /path/to/your/motion.npz \
    --mode 3d \
    --animate \
    --save_gif output.gif

# 2D侧视图
python scripts/visualize_smplx_stickfigure.py \
    --smplx_file /path/to/your/motion.npz \
    --mode 2d \
    --animate
```

### 方法3：在 Python 代码中使用

```python
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from visualize_smplx_simple import visualize_frames_interactive

# 加载数据
smplx_data, body_model, smplx_output, _ = load_smplx_file(
    "your_file.npz", 
    "assets/body_models"
)

# 处理帧
smplx_data_frames, fps = get_smplx_data_offline_fast(
    smplx_data, body_model, smplx_output, tgt_fps=30
)

# 可视化
visualize_frames_interactive(smplx_data_frames, fps)
```

## 可视化模式说明

### Interactive（交互式）- **推荐**
- 底部有滑块，可以手动控制播放哪一帧
- 适合仔细检查特定帧的姿态
- 窗口保持打开，可以来回拖动

### Animation（动画）
- 自动循环播放动画
- 可以保存为 GIF
- 实时播放整个动作序列

### Grid（网格）
- 一次显示多个关键帧
- 快速预览整个动作的关键姿态
- 适合截图或对比

## 示例

```bash
# 最简单的使用方式 - 交互式查看
cd /home/amax/devel/GMR
python scripts/smplx_to_robot.py \
    --smplx_file motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz \
    --robot unitree_g1 \
    --visualize_smplx
```

## 特性

✅ 3D 火柴人可视化
✅ 显示所有主要关节和骨骼连接
✅ 交互式帧控制
✅ 支持动画播放
✅ 可保存为 GIF
✅ 2D 侧视图选项
✅ 关键帧网格显示

## 骨骼连接
可视化包含以下骨骼连接：
- 脊柱：pelvis → spine1 → spine2 → spine3 → neck → head
- 左臂：spine3 → left_collar → left_shoulder → left_elbow → left_wrist
- 右臂：spine3 → right_collar → right_shoulder → right_elbow → right_wrist
- 左腿：pelvis → left_hip → left_knee → left_ankle
- 右腿：pelvis → right_hip → right_knee → right_ankle

## 故障排除

### 如果提示缺少 matplotlib
```bash
pip install matplotlib
```

### 如果图形不显示
确保你的环境支持 GUI 显示，或者使用 `--save_gif` 选项保存到文件。

### 如果想修改关节连接
编辑 `visualize_smplx_simple.py` 中的 `SKELETON_CONNECTIONS` 列表。

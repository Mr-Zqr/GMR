import numpy as np

"""
演示如何从 .npz 文件中索引嵌套的数据

假设 trackings 的结构如下：
trackings: ndarray (object), shape=(1,) 或 shape=()
  └─ 包含一个字典，字典的键包括：
      - smpl_trans_wd: ndarray, shape=(N, 3)
      - smpl_orient: ndarray, shape=(N, 3)
      - 等等...
"""

# 加载数据
data = np.load("/home/amax/devel/dataset/scene_0_426.npz", allow_pickle=True)

print("=" * 70)
print("方法1: 使用 .item() 方法 (推荐用于标量对象数组)")
print("=" * 70)

if 'trackings' in data.files:
    # 方法1: 如果 trackings 是标量对象数组 (shape=())
    try:
        tracking_dict = data['trackings'].item()
        if 'smpl_trans_wd' in tracking_dict:
            smpl_trans_wd = tracking_dict['smpl_trans_wd']
            print(f"✓ smpl_trans_wd 维度: {smpl_trans_wd.shape}")
            print(f"✓ 数据类型: {smpl_trans_wd.dtype}")
        else:
            print(f"可用的键: {list(tracking_dict.keys())}")
    except (ValueError, AttributeError) as e:
        print(f"✗ 方法1失败: {e}")

    print("\n" + "=" * 70)
    print("方法2: 使用索引 [()] (适用于标量数组)")
    print("=" * 70)
    
    # 方法2: 使用 [()] 索引标量数组
    try:
        tracking_dict = data['trackings'][()]
        if isinstance(tracking_dict, dict) and 'smpl_trans_wd' in tracking_dict:
            smpl_trans_wd = tracking_dict['smpl_trans_wd']
            print(f"✓ smpl_trans_wd 维度: {smpl_trans_wd.shape}")
        else:
            print(f"类型: {type(tracking_dict)}")
    except (IndexError, TypeError) as e:
        print(f"✗ 方法2失败: {e}")

    print("\n" + "=" * 70)
    print("方法3: 使用数组索引 [0] (适用于一维数组)")
    print("=" * 70)
    
    # 方法3: 如果 trackings 的 shape=(1,) 或 (N,)
    try:
        if data['trackings'].shape == (1,) or len(data['trackings'].shape) == 1:
            tracking_dict = data['trackings'][0]
            if isinstance(tracking_dict, dict) and 'smpl_trans_wd' in tracking_dict:
                smpl_trans_wd = tracking_dict['smpl_trans_wd']
                print(f"✓ smpl_trans_wd 维度: {smpl_trans_wd.shape}")
            else:
                print(f"类型: {type(tracking_dict)}")
        else:
            print(f"trackings shape 不是 (1,): {data['trackings'].shape}")
    except (IndexError, TypeError) as e:
        print(f"✗ 方法3失败: {e}")

    print("\n" + "=" * 70)
    print("通用访问模式")
    print("=" * 70)
    
    trackings = data['trackings']
    print(f"trackings shape: {trackings.shape}")
    print(f"trackings dtype: {trackings.dtype}")
    
    # 自动选择合适的方法
    if trackings.dtype == object:
        if trackings.shape == ():
            # 标量对象数组
            tracking_dict = trackings.item()
            print(f"\n使用 .item() 方法")
        elif len(trackings.shape) == 1 and trackings.shape[0] == 1:
            # shape=(1,) 的数组
            tracking_dict = trackings[0]
            print(f"\n使用 [0] 索引")
        else:
            # 多元素数组
            tracking_dict = trackings[0]  # 访问第一个元素
            print(f"\n使用 [0] 索引访问第一个元素，共 {trackings.shape[0]} 个元素")
        
        if isinstance(tracking_dict, dict):
            print(f"字典包含的键: {list(tracking_dict.keys())}")
            
            # 访问 smpl_trans_wd
            if 'smpl_trans_wd' in tracking_dict:
                smpl_trans_wd = tracking_dict['smpl_trans_wd']
                print(f"\n★ 成功获取 smpl_trans_wd:")
                print(f"   维度: {smpl_trans_wd.shape}")
                print(f"   数据类型: {smpl_trans_wd.dtype}")
                if isinstance(smpl_trans_wd, np.ndarray) and smpl_trans_wd.size > 0:
                    print(f"   值范围: [{smpl_trans_wd.min():.4f}, {smpl_trans_wd.max():.4f}]")
                    print(f"   前3个值:\n{smpl_trans_wd[:3]}")
else:
    print("'trackings' 不在数据文件中")
    print(f"可用的键: {data.files}")

print("\n" + "=" * 70)
print("总结：推荐的访问方式")
print("=" * 70)
print("""
# 通用方法（自动适配）:
trackings = data['trackings']
if trackings.shape == ():
    smpl_trans_wd = trackings.item()['smpl_trans_wd']
else:
    smpl_trans_wd = trackings[0]['smpl_trans_wd']

# 或者使用 [()] (适用于大多数情况):
smpl_trans_wd = data['trackings'][()]['smpl_trans_wd']
""")

import numpy as np

def print_nested_structure(obj, indent=0, max_depth=5, current_depth=0):
    """递归打印嵌套数据结构"""
    prefix = "  " * indent
    
    if current_depth >= max_depth:
        print(f"{prefix}[达到最大深度限制]")
        return
    
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # 对象数组，可能包含嵌套结构
            print(f"{prefix}类型: ndarray (object)")
            print(f"{prefix}维度: {obj.shape}")
            print(f"{prefix}元素总数: {obj.size}")
            
            # 尝试解析内容
            if obj.size == 0:
                print(f"{prefix}内容: [空数组]")
            elif obj.size == 1:
                # 标量对象数组
                item = obj.item()
                print(f"{prefix}内容类型: {type(item)}")
                if isinstance(item, dict):
                    print(f"{prefix}字典键: {list(item.keys())}")
                    for k, v in item.items():
                        print(f"{prefix}  [{k}]:")
                        print_nested_structure(v, indent + 2, max_depth, current_depth + 1)
                elif isinstance(item, (list, tuple)):
                    print(f"{prefix}列表/元组长度: {len(item)}")
                    for i, v in enumerate(item[:3]):  # 只显示前3个
                        print(f"{prefix}  [索引 {i}]:")
                        print_nested_structure(v, indent + 2, max_depth, current_depth + 1)
                    if len(item) > 3:
                        print(f"{prefix}  ... (还有 {len(item) - 3} 个元素)")
                elif item is None:
                    print(f"{prefix}内容: None")
                else:
                    print(f"{prefix}内容: {item}")
            else:
                # 多元素对象数组
                print(f"{prefix}前几个元素的类型:")
                for i in range(min(3, obj.size)):
                    item = obj.flat[i]
                    print(f"{prefix}  [索引 {i}] 类型: {type(item)}")
                    if isinstance(item, dict):
                        print(f"{prefix}    字典键: {list(item.keys())}")
                        for k, v in item.items():
                            print(f"{prefix}      [{k}]:")
                            print_nested_structure(v, indent + 4, max_depth, current_depth + 1)
                    elif isinstance(item, (list, tuple)):
                        print(f"{prefix}    列表/元组长度: {len(item)}")
                    elif isinstance(item, np.ndarray):
                        print(f"{prefix}    数组维度: {item.shape}, dtype: {item.dtype}")
                    else:
                        print(f"{prefix}    值: {item}")
                if obj.size > 3:
                    print(f"{prefix}  ... (还有 {obj.size - 3} 个元素)")
        else:
            # 普通数值数组
            print(f"{prefix}类型: ndarray")
            print(f"{prefix}数据类型: {obj.dtype}")
            print(f"{prefix}维度: {obj.shape}")
            print(f"{prefix}元素总数: {obj.size}")
            # 显示一些值
            # if obj.size <= 10:
            #     print(f"{prefix}内容: {obj}")
            # else:
            #     print(f"{prefix}值范围: [{obj.min()}, {obj.max()}]")
            #     print(f"{prefix}前几个值: {obj[:]}...")
    
    elif isinstance(obj, dict):
        print(f"{prefix}类型: dict")
        print(f"{prefix}键数量: {len(obj)}")
        print(f"{prefix}键: {list(obj.keys())}")
        for k, v in obj.items():
            print(f"{prefix}  [{k}]:")
            print_nested_structure(v, indent + 2, max_depth, current_depth + 1)
    
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}类型: {type(obj).__name__}")
        print(f"{prefix}长度: {len(obj)}")
        for i, item in enumerate(obj[:3]):
            print(f"{prefix}  [索引 {i}]:")
            print_nested_structure(item, indent + 2, max_depth, current_depth + 1)
        if len(obj) > 3:
            print(f"{prefix}  ... (还有 {len(obj) - 3} 个元素)")
    
    else:
        print(f"{prefix}类型: {type(obj)}")
        print(f"{prefix}值: {obj}")


# 读取 .npy 或 .npz 文件
# data = np.load("/home/amax/devel/dataset/scene_0_426.npz", allow_pickle=True)
# file_path = "/home/amax/devel/dataset/PHUMA/data/g1/aist/subset_0000/Dance_Break_3_Step_clip_10_chunk_0000.npy"
# file_path = "/home/amax/devel/dataset/yiheng_g1/Archive 4/motion.npz"
file_path = "/home/amax/devel/dataset/motion_0-19916_seg0_with_contact.npz"


data = np.load(file_path, allow_pickle=True)

# 查看数据类型
print(f"文件路径: {file_path}")
print(f"数据类型: {type(data)}")
print()

# 检查是 .npz 还是 .npy 文件
if isinstance(data, np.lib.npyio.NpzFile):
    # .npz 文件 - 包含多个数组的字典
    print("=" * 50)
    print("NPZ 文件包含的所有 key:")
    print("=" * 50)
    for key in data.files:
        print(f"- {key}")
    print()

    # 查看每个 key 的维度和详细信息
    print("=" * 70)
    print("每个 key 的详细信息 (包括嵌套结构):")
    print("=" * 70)
    for key in data.files:
        value = data[key]
        print(f"\n{'='*70}")
        print(f"Key: {key}")
        print(f"{'='*70}")
        print_nested_structure(value, indent=1)
else:
    # .npy 文件 - 单个数组
    print("=" * 70)
    print("NPY 文件内容 (单个数组):")
    print("=" * 70)
    print_nested_structure(data, indent=0)

# 演示如何索引 trackings 下的 smpl_trans_wd
print("\n" + "=" * 70)
print("示例：如何索引 trackings 下的数据")
print("=" * 70)

# 仅对 .npz 文件执行此操作
if isinstance(data, np.lib.npyio.NpzFile) and 'trackings' in data.files:
    trackings = data['trackings']
    print(f"\ntrackings 类型: {type(trackings)}")
    print(f"trackings 维度: {trackings.shape}")
    print(f"trackings dtype: {trackings.dtype}")
    
    if trackings.dtype == object and trackings.size > 0:
        # trackings 是一个 object 数组，通常包含字典
        # 如果是标量数组 (维度为 () 或 (1,))
        if trackings.shape == () or (len(trackings.shape) == 1 and trackings.shape[0] == 1):
            tracking_dict = trackings.item() if trackings.shape == () else trackings[0]
            print(f"\ntracking 字典的键: {list(tracking_dict.keys()) if isinstance(tracking_dict, dict) else '不是字典'}")
            
            if isinstance(tracking_dict, dict) and 'smpl_trans_wd' in tracking_dict:
                smpl_trans_wd = tracking_dict['smpl_trans_wd']
                print(f"\n✓ 成功访问 smpl_trans_wd!")
                print(f"  类型: {type(smpl_trans_wd)}")
                if isinstance(smpl_trans_wd, np.ndarray):
                    print(f"  维度: {smpl_trans_wd.shape}")
                    print(f"  数据类型: {smpl_trans_wd.dtype}")
                    print(f"  值范围: [{smpl_trans_wd.min()}, {smpl_trans_wd.max()}]")
                    print(f"\n  索引方式:")
                    print(f"    data['trackings'].item()['smpl_trans_wd']")
                    print(f"    或者: data['trackings'][()]['smpl_trans_wd']")
            else:
                print(f"\n✗ 'smpl_trans_wd' 不在 tracking 字典中")
                if isinstance(tracking_dict, dict):
                    print(f"  可用的键: {list(tracking_dict.keys())}")
        else:
            # 多元素数组
            print(f"\ntrackings 包含 {trackings.size} 个元素")
            print(f"第一个元素类型: {type(trackings[0])}")
            if isinstance(trackings[0], dict):
                print(f"第一个元素的键: {list(trackings[0].keys())}")
                if 'smpl_trans_wd' in trackings[0]:
                    smpl_trans_wd = trackings[0]['smpl_trans_wd']
                    print(f"\n✓ 成功访问 smpl_trans_wd!")
                    print(f"  类型: {type(smpl_trans_wd)}")
                    if isinstance(smpl_trans_wd, np.ndarray):
                        print(f"  维度: {smpl_trans_wd.shape}")
                        print(f"  数据类型: {smpl_trans_wd.dtype}")
                        print(f"\n  索引方式:")
                        print(f"    data['trackings'][0]['smpl_trans_wd']")
    else:
        print("\ntrackings 不是 object 类型或为空")
elif isinstance(data, np.lib.npyio.NpzFile):
    print("\n✗ 'trackings' 不在 npz 文件中")
    print(f"可用的键: {data.files}")
else:
    print("\n这是一个 .npy 文件，包含单个数组，不是 .npz 字典格式")
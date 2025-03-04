import os
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
current_dir = os.path.dirname(os.path.abspath(__file__))
motion_dir = os.path.join(current_dir, "../envs/x02_amp/datasets")
npy_files = [f for f in os.listdir(motion_dir) if f.endswith(".npy")]
for npy_file in npy_files:
    # 构建完整的文件路径
    npy_path = os.path.join(motion_dir, npy_file)

    # 加载 .npy 文件
    data = np.load(npy_path, allow_pickle=True)
    if data.ndim == 0:  # 检查是否是 0D 数组
        data = np.array([data])  # 转换为 1D 数组
    # 构建对应的 .txt 文件路径
    txt_file = npy_file.replace(".npy", ".txt")
    txt_path = os.path.join(motion_dir, txt_file)

    # 将数据保存为 .txt 文件
    np.savetxt(txt_path, data, fmt="%s")  # fmt 参数控制数据格式

    print(f"Converted {npy_file} to {txt_file}")
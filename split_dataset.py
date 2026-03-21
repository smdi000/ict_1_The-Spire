# split_dataset.py
import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_ultra_dataset(
    input_path="./data/ultra_train.npz",   # 原始全量数据
    output_dir="./data",
    val_ratio=0.2,
    random_seed=42
):
    print(f"📂 Loading full dataset from: {input_path}")
    data = np.load(input_path)
    
    X = data['X']          # (N, 39, 50)
    Y_cls = data['Y_cls']  # (N,)
    Y_traj = data['Y_traj']# (N, 72)
    
    N = len(X)
    print(f"📊 Total samples: {N}")
    
    # 划分索引（按样本划分，保持 X/Y 对齐）
    indices = np.arange(N)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=Y_cls  # 保证分类标签分布一致（可选，若类别平衡可去掉）
    )
    
    print(f"✂️  Splitting into:")
    print(f"   Train: {len(train_idx)} samples ({100*(1-val_ratio):.1f}%)")
    print(f"   Val  : {len(val_idx)} samples ({100*val_ratio:.1f}%)")
    
    # 保存训练集
    train_path = os.path.join(output_dir, "ultra_train.npz")
    np.savez_compressed(
        train_path,
        X=X[train_idx],
        Y_cls=Y_cls[train_idx],
        Y_traj=Y_traj[train_idx]
    )
    print(f"✅ Saved train set to: {train_path}")
    
    # 保存验证集
    val_path = os.path.join(output_dir, "ultra_val.npz")
    np.savez_compressed(
        val_path,
        X=X[val_idx],
        Y_cls=Y_cls[val_idx],
        Y_traj=Y_traj[val_idx]
    )
    print(f"✅ Saved val set to: {val_path}")
    
    # 👇 关键：只用训练集计算标准化参数！
    traj_mean = Y_traj[train_idx].mean(axis=0)
    traj_std = Y_traj[train_idx].std(axis=0) + 1e-8
    
    stats_path = os.path.join(output_dir, "traj_stats.npz")
    np.savez(stats_path, mean=traj_mean, std=traj_std)
    print(f"✅ Saved normalization stats (from TRAIN ONLY) to: {stats_path}")
    print(f"   Mean[0]: {traj_mean[0]:.2f}, Std[0]: {traj_std[0]:.2f}")

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    split_ultra_dataset(
        input_path="./data/ultra_train.npz",  # 假设这是你的全量数据
        output_dir="./data",
        val_ratio=0.2,
        random_seed=42
    )
# inspect_h5.py
import h5py
import sys

def print_h5_structure(name, obj):
    """递归打印 HDF5 结构"""
    if isinstance(obj, h5py.Dataset):
        print(f"DATASET: {name} → shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"GROUP:   {name}")

if __name__ == "__main__":
    h5_path = "All_subjects_data.h5"  # 👈 替换为你的实际路径
    with h5py.File(h5_path, 'r') as f:
        print("🔍 HDF5 文件完整结构：")
        f.visititems(print_h5_structure)
# hprocess.py —— 支持 108/111/112 维
import h5py
import numpy as np

EXERCISE_MAP = {"AS": 0, "CB": 1, "EF": 2, "ER": 3, "OR": 4}

def safe_convert_to_float(value):
    if isinstance(value, bytes):
        if value == b'':
            return np.nan
        try:
            return float(value.decode('utf-8'))
        except (UnicodeDecodeError, ValueError):
            return np.nan
    elif isinstance(value, str):
        if value == '':
            return np.nan
        try:
            return float(value)
        except ValueError:
            return np.nan
    elif isinstance(value, (int, float, np.number)):
        return float(value)
    else:
        return np.nan

def main():
    H5_PATH = "data\All_subjects_data.h5"
    WINDOW_LEN = 50
    STRIDE = 10
    TRAJ_DIM = 72  # 固定：9 joints × 8 future steps

    X_list, Y_cls_list, Y_traj_list = [], [], []

    with h5py.File(H5_PATH, 'r') as f:
        for subject in f.keys():
            print(f"Processing {subject}...")
            for exercise in f[subject].keys():
                if exercise not in EXERCISE_MAP:
                    continue
                label = EXERCISE_MAP[exercise]
                for speed in f[f"{subject}/{exercise}"].keys():
                    raw_data = f[f"{subject}/{exercise}/{speed}"][:]
                    T, D = raw_data.shape
                    print(f"  {exercise}/{speed}: shape=({T}, {D})")

                    # 安全转换
                    data = np.full((T, D), np.nan, dtype=np.float32)
                    for i in range(T):
                        for j in range(D):
                            data[i, j] = safe_convert_to_float(raw_data[i, j])
                    
                    if np.isnan(data).all():
                        print(f"    ❌ All NaN! Skipping.")
                        continue

                    # 用列均值填充 NaN
                    col_means = np.nanmean(data, axis=0)
                    nan_mask = np.isnan(data)
                    data[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

                    # === 动态处理维度 ===
                    if D == 112:
                        data = data[:, :-1]  # 去掉时间戳
                        D = 111
                    elif D not in [108, 111]:
                        print(f"    ⚠️ Unsupported dim {D}, skipping")
                        continue

                    # 关键：轨迹始终是最后 TRAJ_DIM=72 维
                    if D < TRAJ_DIM:
                        print(f"    ⚠️ D={D} < TRAJ_DIM={TRAJ_DIM}, skipping")
                        continue

                    sensor_dim = D - TRAJ_DIM  # 111-72=39, 108-72=36
                    sensors = data[:, :sensor_dim]      # (T, 36 or 39)
                    traj_gt = data[:, -TRAJ_DIM:]       # (T, 72)

                    # 统一传感器维度到 39（对108维数据补0）
                    if sensor_dim == 36:
                        # 补3个零通道（假设缺失的是IMU后3通道）
                        sensors = np.pad(sensors, ((0, 0), (0, 3)), mode='constant')
                    elif sensor_dim != 39:
                        print(f"    ⚠️ Unexpected sensor_dim={sensor_dim}, skipping")
                        continue

                    if T < WINDOW_LEN:
                        continue

                    for start in range(0, T - WINDOW_LEN, STRIDE):
                        x = sensors[start:start+WINDOW_LEN].T.astype(np.float32)  # (39, 50)
                        y_cls = np.array(label, dtype=np.int32)
                        y_traj = traj_gt[start].astype(np.float32)

                        X_list.append(x)
                        Y_cls_list.append(y_cls)
                        Y_traj_list.append(y_traj)

    if not X_list:
        raise RuntimeError("No valid data!")

    X = np.stack(X_list)
    Y_cls = np.stack(Y_cls_list)
    Y_traj = np.stack(Y_traj_list)

    np.savez_compressed("ultra_train.npz", X=X, Y_cls=Y_cls, Y_traj=Y_traj)
    print(f"✅ Success! Total samples: {len(X)}")
    print(f"   X shape: {X.shape}")
    print(f"   Sensor dim: {X.shape[1]} (should be 39)")

if __name__ == "__main__":
    main()
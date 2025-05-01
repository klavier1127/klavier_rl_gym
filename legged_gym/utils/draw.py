import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = '/home/droid/IssacGym-projects/klavier_rl_gym/legged_gym/utils/wandb_export_2025-04-27T09_50_36.094+08_00.csv'
df = pd.read_csv(file_path)

# 构造训练轮次 (Iteration) 序列
x_full = pd.Series(df.index + 1)
max_iter = len(df)

# 平滑窗口大小
window_size = 500

# 筛选三条主曲线
cols = [c for c in df.columns if 'Episode/terrain_level' in c and '__MIN' not in c and '__MAX' not in c]
lpd_col, ts_col, baseline_col = cols

# 滑动平均平滑
sm = df[cols].rolling(window=window_size, min_periods=1).mean()

# 横向平移 TS 曲线到第 3000 轮开始
x_ts = x_full# + 2999
mask_ts = x_ts <= max_iter
x_ts = x_ts[mask_ts].reset_index(drop=True)
ts_vals = sm[ts_col][mask_ts].reset_index(drop=True)

# 裁剪长度设置
clip_lpd = 520        # LPD 前 520 个 Iteration 置为 0
clip_ts = 520         # TS 前 520 个 Iteration 置为 0
clip_baseline = 370   # Baseline 前 370 个 Iteration 置为 0

# 生成绘图数据
lpd_plot = sm[lpd_col].copy()
lpd_plot.iloc[:clip_lpd] = 0

ts_plot = ts_vals.copy()
ts_plot[x_ts < (x_ts.min() + clip_ts)] = 0

baseline_plot = sm[baseline_col].copy()
baseline_plot.iloc[:clip_baseline] = 0

# 绘图
plt.figure(figsize=(8,5))
plt.plot(x_full, lpd_plot, label='LPD', color='red')
plt.plot(x_ts, ts_plot, label='T-S', color='green')
plt.plot(x_full, baseline_plot, label='Baseline', color='blue')

plt.xlim(1, max_iter)
plt.xlabel('Iteration')
plt.ylabel('Terrain Level')
plt.legend(loc='lower right', fontsize=14)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# 从 CSV 文件读取两列数据
values1 = []
values2 = []
with open('../actions_log.csv', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        try:
            v1, v2 = float(parts[0]), float(parts[1])
            values1.append(v1)
            values2.append(v2)
        except:
            continue

# 创建带 Axes 对象的 Figure
fig, ax = plt.subplots(figsize=(10, 5))

# 画背景色块（zorder=0 保证在最底层）
ax.axvspan(   0,  160, facecolor='purple', alpha=0.2, zorder=0)
ax.axvspan( 160,  550, facecolor='orange', alpha=0.2, zorder=0)
ax.axvspan( 550,  900, facecolor='skyblue', alpha=0.2, zorder=0)

# 2) 在每段背景下方添加文字标签
for label, start, end, color in [
    ('plane',     0, 160, 'purple'),
    ('stair up',160, 560, 'orange'),
    ('stair down',560, 900, 'skyblue'),
]:
    x_text = (start + end) / 2
    ax.text(
        x_text,  # 水平位置：区间中点
        0.02,  # 垂直位置：比 x 轴稍微高一点（以坐标轴高度的百分比为单位）
        label,
        transform=ax.get_xaxis_transform(),  # x 按数据，y 按轴比例
        ha='center', va='bottom',  # 底部对齐，让文字紧贴在 x 轴上方
        fontsize=14,
        color=color
    )


# 再画两条折线，zorder>0 保证在色块之上
ax.plot(values1, label='student actions', linewidth=1, zorder=1)
ax.plot(values2, label='teacher actions', linewidth=1, zorder=1)

# 设置标题、坐标轴和图例
ax.set_xlabel('Timestep', fontsize=16)
ax.set_ylabel('L2 Norm of Action', fontsize=16)
# 如果你不想在 legend 中重复色块标签，可以只传折线的句柄
# lines, labels = ax.get_legend_handles_labels()
# ax.legend(lines, labels, fontsize=14)
ax.legend(fontsize=14, loc='upper right')

ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

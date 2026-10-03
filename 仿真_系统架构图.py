# -*- coding: utf-8 -*-
"""
系统架构图（Systems 版 Figure 1）
动态社会认知系统：跨文化意义建构的最小计算系统架构
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(7.6, 9.2))
ax.set_xlim(0, 10)
ax.set_ylim(-0.6, 13)
ax.axis("off")

C_TOP = "#e8f0fe"; E_TOP = "#1a73e8"     # 蓝色：符号输入
C_NOV = "#fce8e6"; E_NOV = "#d93025"     # 红：新奇/感知
C_SEM = "#e6f4ea"; E_SEM = "#188038"     # 绿：语义
C_STA = "#fef7e0"; E_STA = "#f9ab00"     # 黄：接地状态
C_APP = "#f3e8fd"; E_APP = "#9334e6"     # 紫：欣赏
C_REV = "#f1f3f4"; E_REV = "#5f6368"     # 灰：涌现反转

def box(x, y, w, h, text, fc, ec, fs=11, lw=1.6):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)

def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0, color="#444444"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                connectionstyle="arc3,rad=0"))
    if label:
        ax.text(lx, ly, label, fontsize=9, color="#444444",
                ha="center", va="center")

# ---- 1. 文化符号（输入） ----
box(2.0, 11.5, 6.0, 1.2, "Cultural Symbol\n(Signifier + Signified)", C_TOP, E_TOP, fs=11.5)

# ---- 2. 感知层 / 语义层（并列） ----
box(0.5, 9.0, 4.3, 1.5, "Perceptual Layer\nNovelty  $V_N$", C_NOV, E_NOV)
box(5.2, 9.0, 4.3, 1.5, "Semantic Layer\nMeaning  $V_M$", C_SEM, E_SEM)

# ---- 3. 接地状态（系统状态变量） ----
box(2.0, 6.3, 6.0, 1.3, "Grounding State  $g \\in [0,1]$\n(system-state variable)", C_STA, E_STA, fs=11)

# ---- 4. 动态重加权 ----
box(0.8, 3.9, 3.6, 1.4, "Novelty weight\n$w(g)$", C_NOV, E_NOV)
box(5.6, 3.9, 3.6, 1.4, "Meaning weight\n$1 - w(g)$", C_SEM, E_SEM)

# ---- 5. 欣赏 ----
box(1.2, 1.6, 7.6, 1.3, "Appreciation  $A(g) = w(g)\\,V_N + (1-w(g))\\,V_M$", C_APP, E_APP, fs=11.5)

# ---- 6. 涌现反转 ----
box(2.5, -0.1, 5.0, 1.1, "Preference-order Reversal\n($D(g)=0$, emergent)", C_REV, E_REV, fs=10.5)

# ---- 箭头 ----
arrow(3.7, 11.5, 2.65, 10.5)                # 符号 -> 感知层
arrow(6.3, 11.5, 7.35, 10.5)                # 符号 -> 语义层
arrow(2.65, 9.0, 3.6, 7.6)                  # 感知层 -> 接地状态
arrow(7.35, 9.0, 6.4, 7.6)                  # 语义层 -> 接地状态
ax.text(5.0, 8.25, "grounding", fontsize=9, color="#444444", ha="center", va="center")

arrow(3.8, 6.3, 2.6, 5.3)                   # 接地 -> wN
arrow(6.2, 6.3, 7.4, 5.3)                   # 接地 -> 1-wN
ax.text(5.0, 5.55, "dynamic\nreweighting", fontsize=9, color="#444444", ha="center", va="center")

arrow(2.6, 3.9, 3.9, 2.9)                   # wN -> 欣赏
arrow(7.4, 3.9, 6.1, 2.9)                   # 1-wN -> 欣赏
arrow(5.0, 1.6, 5.0, 1.0)                   # 欣赏 -> 反转
ax.text(5.7, 1.28, "emergent", fontsize=8.5, color="#666666", ha="left", va="center")

fig.tight_layout()
out = "仿真_系统架构图.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[OK] 系统架构图已保存：%s" % out)

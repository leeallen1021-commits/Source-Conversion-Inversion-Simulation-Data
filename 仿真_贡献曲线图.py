# -*- coding: utf-8 -*-
"""
贡献曲线图：展示"来源主导转变(g_s)"与"偏好反转(g*)"是两个不同的系统事件
Type N 与 Type M 各画新奇贡献 C_novelty(g) 与语义贡献 C_meaning(g)，
交叉点即来源主导转变点 g_s；灰色竖虚线为共享的偏好反转点 g*。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Vn_N, Vm_N = 6.5, 3.0   # Type N：高新奇、低语义
Vn_M, Vm_M = 3.5, 6.5   # Type M：低新奇、高语义
k, g0 = 8.0, 0.5

g = np.linspace(0.0, 1.0, 400)
w = 1.0 / (1.0 + np.exp(k * (g - g0)))

def g_of_w(wstar):
    return g0 + (1.0 / k) * np.log(1.0 / wstar - 1.0)

# 三个阈值
g_rev = g0 - (1.0 / k) * np.log((Vm_M - Vm_N) / (Vn_N - Vn_M))   # 偏好反转点
g_sN = g_of_w(Vm_N / (Vn_N + Vm_N))                              # Type N 来源转变
g_sM = g_of_w(Vm_M / (Vn_M + Vm_M))                              # Type M 来源转变

# 贡献曲线
C_N_nov = w * Vn_N
C_N_mea = (1 - w) * Vm_N
C_M_nov = w * Vn_M
C_M_mea = (1 - w) * Vm_M

fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.4), sharex=True)

# ---------- 上：Type N ----------
ax = axes[0]
ax.plot(g, C_N_nov, color="#d62728", lw=2.2, label="$C_{novelty}(g)$")
ax.plot(g, C_N_mea, color="#1f77b4", lw=2.2, label="$C_{meaning}(g)$")
y_cross = w[int(g_sN * 400)] * Vn_N
ax.axvline(g_rev, color="gray", ls="--", lw=1.3, alpha=0.8)
ax.axvline(g_sN, color="black", ls=":", lw=1.3)
ax.scatter([g_sN], [y_cross], color="black", zorder=5, s=55)
ax.annotate("source shift\n$g_s(N)\\approx 0.60$", xy=(g_sN, y_cross),
            xytext=(g_sN + 0.13, y_cross + 0.7), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate("reversal\n$g^*\\approx 0.48$", xy=(g_rev, 5.2),
            xytext=(g_rev - 0.30, 5.4), fontsize=9, color="gray",
            arrowprops=dict(arrowstyle="->", lw=1, color="gray"))
ax.set_ylabel("contribution", fontsize=10)
ax.set_ylim(0, 7.0)
ax.legend(loc="center left", fontsize=9)
ax.grid(alpha=0.25)
ax.set_title("Type N (high novelty, low meaning)", fontsize=10.5)

# ---------- 下：Type M ----------
ax = axes[1]
ax.plot(g, C_M_nov, color="#d62728", lw=2.2, label="$C_{novelty}(g)$")
ax.plot(g, C_M_mea, color="#1f77b4", lw=2.2, label="$C_{meaning}(g)$")
y_cross = w[int(g_sM * 400)] * Vn_M
ax.axvline(g_rev, color="gray", ls="--", lw=1.3, alpha=0.8)
ax.axvline(g_sM, color="black", ls=":", lw=1.3)
ax.scatter([g_sM], [y_cross], color="black", zorder=5, s=55)
ax.annotate("source shift\n$g_s(M)\\approx 0.42$", xy=(g_sM, y_cross),
            xytext=(g_sM - 0.05, y_cross + 1.2), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate("reversal\n$g^*\\approx 0.48$", xy=(g_rev, 5.2),
            xytext=(g_rev + 0.08, 5.5), fontsize=9, color="gray",
            arrowprops=dict(arrowstyle="->", lw=1, color="gray"))
ax.set_xlabel("grounding depth g", fontsize=10)
ax.set_ylabel("contribution", fontsize=10)
ax.set_ylim(0, 7.0)
ax.legend(loc="center right", fontsize=9)
ax.grid(alpha=0.25)
ax.set_title("Type M (low novelty, high meaning)", fontsize=10.5)

fig.tight_layout()
out = "仿真_贡献曲线图.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[OK] 贡献曲线图已保存：%s" % out)
print("[INFO] g_s(N)=%.3f  g*=%.3f  g_s(M)=%.3f" % (g_sN, g_rev, g_sM))

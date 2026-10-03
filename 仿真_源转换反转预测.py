# -*- coding: utf-8 -*-
"""
源转换（Source Shift）双分量加权模型 · 三个计算实验
研究② 符号理解机制 · 纯仿真版（B 形态：建模 + 预注册实证方案）

模型：
    Appreciation(g) = w(g) * V_novelty + (1 - w(g)) * V_meaning      (Eq.1)
    w(g) = 1 / (1 + exp(k * (g - g0)))                              (Eq.2)

三个计算实验（各输出一张独立图）：
    Exp1 生成充分性  —— 主模型扫描 g，画 N/M 偏好线，证明能生成"偏好反转"
    Exp2 对比模型检验 —— M1-M4（权重恒定）不反转；主模型与 M5（新奇不对称衰减）
                        均反转 → 反转的必要条件是"新奇有效权重随 g 下降"，
                        语义接地是含独立语义项的可操作化实现（对应判别预测 T5）
    Exp3 敏感性      —— 扫描 (k, g0) 参数空间，画"反转/不反转"相图

诚实声明：所有输出均为 model-generated（synthetic）数据，不涉及经验数据。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 参数（默认，仅用于生成理论预测形态） ----------
k0 = 8.0        # 默认翻转陡峭度
g0_0 = 0.5      # 默认翻转中点

# 符号价值设定（Type N = 新奇主导，Type M = 语义主导）
Vn_N, Vm_N = 6.5, 3.0   # Type N：高新奇、低语义
Vn_M, Vm_M = 3.5, 6.5   # Type M：低新奇、高语义

dN = Vn_N - Vn_M        # 新奇价值差
dM = Vm_M - Vm_N        # 语义价值差
w_star = dM / (dN + dM) # 反转处的接地权重

# ---------- 模型函数 ----------
def w(g, k=k0, g0=g0_0):
    """接地权重：随接地深度从 1（纯新奇）单调降到 0（纯语义）"""
    return 1.0 / (1.0 + np.exp(k * (g - g0)))

def appreciation(g, Vn, Vm, k=k0, g0=g0_0):
    """审美欣赏 = 新奇价值与语义价值的加权和"""
    ww = w(g, k, g0)
    return ww * Vn + (1 - ww) * Vm

def g_star(k=k0, g0=g0_0):
    """反转点的解析解（Eq.4）：g* = g0 - (1/k)·ln(ΔV_meaning/ΔV_novelty)"""
    return g0 - (1.0 / k) * np.log(dM / dN)

g = np.linspace(0.0, 1.0, 400)
out_dir = os.path.dirname(os.path.abspath(__file__))

# ================= Exp1 生成充分性 =================
fig1, ax = plt.subplots(figsize=(7.2, 5.0))
app_N = appreciation(g, Vn_N, Vm_N)
app_M = appreciation(g, Vn_M, Vm_M)
gst = g_star()
app_star = appreciation(gst, Vn_N, Vm_N)

ax.plot(g, app_N, color="#d62728", lw=2.5,
        label="Type N (novelty-driven)")
ax.plot(g, app_M, color="#1f77b4", lw=2.5,
        label="Type M (meaning-driven)")
ax.axvline(gst, color="gray", ls="--", lw=1, alpha=0.6)
ax.scatter([gst], [app_star], color="black", zorder=5, s=48)
ax.annotate("crossover\ng* ≈ %.2f" % gst,
            xy=(gst, app_star), xytext=(gst + 0.16, app_star - 1.9),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))
ax.text(0.05, 7.35, "shallow:\nN > M", fontsize=9, ha="left", va="top",
        color="#d62728")
ax.text(0.95, 7.35, "deep:\nM > N", fontsize=9, ha="right", va="top",
        color="#1f77b4")
ax.set_xlabel("grounding depth g (G0 → G3)", fontsize=10)
ax.set_ylabel("appreciation", fontsize=10)
ax.set_xticks([0, .25, .5, .75, 1])
ax.set_xticklabels(["G0", "", "", "", "G3"])
ax.set_ylim(0, 7.6)
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.25)
ax.set_title("Exp1 · Generative Sufficiency\n(the model generates preference reversal)",
             fontsize=11, weight="bold")
fig1.tight_layout()
out1 = os.path.join(out_dir, "仿真_Exp1_生成充分性.png")
fig1.savefig(out1, dpi=200)
plt.close(fig1)
print("[OK] Exp1 图已保存：%s" % out1)

# ================= Exp2 对比模型检验 =================
fig2, ax = plt.subplots(figsize=(7.2, 5.0))
# 偏好差 diff = App_N - App_M；跨 0 即反转
diff_main = appreciation(g, Vn_N, Vm_N) - appreciation(g, Vn_M, Vm_M)
diff_m1 = np.full_like(g, dN)                  # M1 纯新奇 w=1：恒正
diff_m2 = np.full_like(g, -dM)                 # M2 纯语义 w=0：恒负
diff_m3 = np.full_like(g, 0.5 * dN - 0.5 * dM) # M3 固定权重 w=c：恒定
diff_m4 = dN - 0.5 * g                         # M4 纯暴露/流畅：单调降但不变号（恒正）
# M5 新奇不对称衰减：高新颖的 Type N 衰减更快（Berlyne 新奇饱和 / 暴露习惯化）
lam_N, lam_M = 1.5, 0.3
diff_m5 = Vn_N * np.exp(-lam_N * g) - Vn_M * np.exp(-lam_M * g)   # M5 反转（穿零）

ax.plot(g, diff_main, color="black", lw=2.5, label="main model (semantic grounding)")
ax.plot(g, diff_m1, color="#d62728", ls="--", lw=1.4,
        label="M1 · novelty only (w = 1)")
ax.plot(g, diff_m2, color="#1f77b4", ls="--", lw=1.4,
        label="M2 · meaning only (w = 0)")
ax.plot(g, diff_m3, color="#2ca02c", ls="--", lw=1.4,
        label="M3 · fixed weight (w = 0.5)")
ax.plot(g, diff_m4, color="#9467bd", ls="--", lw=1.4,
        label="M4 · exposure / fluency")
ax.plot(g, diff_m5, color="#8c564b", ls="-.", lw=1.8,
        label="M5 · novelty decay (asymmetric)")
ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
ax.set_xlabel("grounding depth g", fontsize=10)
ax.set_ylabel("preference difference (N − M)", fontsize=10)
ax.set_xticks([0, .5, 1])
ax.set_ylim(-4.5, 4.5)
ax.legend(loc="center right", fontsize=8)
ax.grid(alpha=0.25)
ax.set_title("Exp2 · Alternative-model Comparison\n(main & novelty-decay cross 0; weight-constant models do not)",
             fontsize=11, weight="bold")
fig2.tight_layout()
out2 = os.path.join(out_dir, "仿真_Exp2_对比模型检验.png")
fig2.savefig(out2, dpi=200)
plt.close(fig2)
print("[OK] Exp2 图已保存：%s" % out2)

# M5 新奇衰减的穿零点（解析）：g* = ln(Vn_M/Vn_N) / (λ_M − λ_N)
g_star_m5 = np.log(Vn_M / Vn_N) / (lam_M - lam_N)
print("[INFO] Exp2 M5 参数：λN=%.1f, λM=%.1f，穿零点 g*≈%.3f" % (lam_N, lam_M, g_star_m5))

# ================= Exp3 敏感性（相图） =================
fig3, ax = plt.subplots(figsize=(7.2, 5.0))
k_vals = np.linspace(0.5, 20, 200)
g0_vals = np.linspace(0.0, 1.0, 200)
KK, GG = np.meshgrid(k_vals, g0_vals)
# 反转点 g* = g0 - ln(dM/dN)/k；反转 ⟺ 0 < g* < 1
gstar_grid = GG - (1.0 / KK) * np.log(dM / dN)
reversal = (gstar_grid > 0) & (gstar_grid < 1)

ax.pcolormesh(KK, GG, reversal, cmap="RdYlGn", vmin=0, vmax=1, shading="auto")
ax.contour(KK, GG, reversal.astype(float), levels=[0.5],
           colors="black", linewidths=1.2)
ax.scatter([k0], [g0_0], color="black", s=55, zorder=5)
ax.annotate("default\n(k=8, g0=0.5)", xy=(k0, g0_0),
            xytext=(11.5, 0.18), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1))
ax.set_xlabel("k (steepness)", fontsize=10)
ax.set_ylabel("g0 (midpoint)", fontsize=10)
ax.set_title("Exp3 · Sensitivity\n(reversal region in the (k, g0) plane)",
             fontsize=11, weight="bold")
ax.text(0.03, 0.04, "green = reversal\nred = no reversal",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
fig3.tight_layout()
out3 = os.path.join(out_dir, "仿真_Exp3_敏感性.png")
fig3.savefig(out3, dpi=200)
plt.close(fig3)
print("[OK] Exp3 图已保存：%s" % out3)

# ================= Exp4 函数形式稳健性 =================
# 验证反转不依赖 logistic 函数形式：线性/指数/幂函数替代均能反转
fig4, ax = plt.subplots(figsize=(7.2, 5.0))

def D_from_w(wvals):
    """偏好差 = w*dN - (1-w)*dM；跨 0 即反转"""
    return wvals * dN - (1.0 - wvals) * dM

w_log = w(g)                    # 主模型 logistic
w_lin = 1.0 - g                 # 线性
w_exp = np.exp(-2.5 * g)        # 指数衰减（λ=2.5）
w_pow = (1.0 - g) ** 2.0        # 幂函数（α=2）

ax.plot(g, D_from_w(w_log), color="black", lw=2.5,
        label="logistic  w(g)=1/(1+e^{k(g-g0)})")
ax.plot(g, D_from_w(w_lin), color="#d62728", ls="--", lw=1.6,
        label="linear  w(g)=1−g")
ax.plot(g, D_from_w(w_exp), color="#1f77b4", ls="--", lw=1.6,
        label="exponential  w(g)=e^{−λg}")
ax.plot(g, D_from_w(w_pow), color="#2ca02c", ls="--", lw=1.6,
        label="power  w(g)=(1−g)^α")
ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
ax.set_xlabel("grounding depth g", fontsize=10)
ax.set_ylabel("preference difference (N − M)", fontsize=10)
ax.set_xticks([0, .5, 1])
ax.set_ylim(-4.5, 4.5)
ax.legend(loc="center right", fontsize=8)
ax.grid(alpha=0.25)
ax.set_title("Exp4 · Functional-Form Robustness\n(reversal persists across alternative weight functions)",
             fontsize=11, weight="bold")
fig4.tight_layout()
out4 = os.path.join(out_dir, "仿真_Exp4_函数形式稳健性.png")
fig4.savefig(out4, dpi=200)
plt.close(fig4)
print("[OK] Exp4 图已保存：%s" % out4)

# ---------- 汇总 ----------
print("[INFO] Exp1 反转点：g* ≈ %.3f，w* ≈ %.3f" % (gst, w_star))
print("[INFO] 反转存在条件：0 < g* < 1，即 (1/k)·ln(dM/dN) < g0 < 1 + (1/k)·ln(dM/dN)")
print("[INFO] 默认参数下反转边界：%.3f < g0 < %.3f" %
      ((1.0/k0)*np.log(dM/dN), 1 + (1.0/k0)*np.log(dM/dN)))

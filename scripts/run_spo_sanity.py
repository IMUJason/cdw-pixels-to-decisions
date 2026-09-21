"""SPO+ 正向对照（sanity check）：在 Elmachtoub & Grigas (2022) 风格的最短路问题上，
验证 SPO+ 训练实现能复现文献正结果（SPO+ 遗憾 < MSE 遗憾）。

目的：排除"CDW 多物料 LIRP 上 SPO+ 全负结果是实现错误所致"的假设。
若本实验中 SPO+ 显著优于 MSE ⇒ 实现正确，CDW 上的负结果归因于问题结构
（近充分特征 + 分段常数一阶段决策），与论文 6.6 节的机制解释一致。

设定（复刻 Elmachtoub & Grigas 2022 的 grid shortest path 实验家族）：
- 5×5 网格（40 条有向边，DAG 最短路）；
- 真成本生成：c_e = ((B x)_e / sqrt(p) + 3)^deg + 1, deg=4, p=5 特征；
- 预测模型：线性 ĉ = W x（与文献一致）；
- MSE 训练 vs SPO+ 训练（梯度 2(z*(c)-z*(2ĉ-c))，同 spo_multi_trainer）；
- 评估：held-out 测试集上的 normalized SPO regret。

用法: python scripts/run_spo_sanity.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

W_GRID, H_GRID, P_FEAT, DEG = 5, 5, 5, 4
N_EDGES = 2 * W_GRID * (H_GRID - 1)  # 右向 + 下向有向边
SEED = 0


def edges():
    """返回 (edge_list, adj)：edge (u->v)，节点编号 r*W+c。"""
    E, adj = [], {}
    for r in range(H_GRID):
        for c in range(W_GRID):
            u = r * W_GRID + c
            if c + 1 < W_GRID:
                E.append((u, u + 1))
            if r + 1 < H_GRID:
                E.append((u, u + W_GRID))
    for i, (u, v) in enumerate(E):
        adj.setdefault(u, []).append((v, i))
    return E, adj


E, ADJ = edges()


def shortest_path(cost):
    """DAG 最短路（从 0 到 W*H-1），返回路径边指标向量（0/1, len=N_EDGES）。"""
    n = W_GRID * H_GRID
    dist = np.full(n, np.inf)
    dist[0] = 0.0
    prev = {}
    order = sorted(range(n))  # DAG 拓扑序（行优先编号天然合法）
    for u in order:
        for v, i in ADJ.get(u, []):
            if dist[u] + cost[i] < dist[v]:
                dist[v] = dist[u] + cost[i]
                prev[v] = (u, i)
    z = np.zeros(N_EDGES)
    v = n - 1
    while v != 0:
        u, i = prev[v]
        z[i] = 1.0
        v = u
    return z


def gen_data(n, B, rng, noise=0.5):
    X = rng.normal(0, 1, (n, P_FEAT))
    eps = rng.normal(0, 1, (n, N_EDGES))
    # 带噪成本（文献设定）：不可约误差使 MSE 不再完美可学，SPO+ 的决策对齐才有空间
    C = ((X @ B.T + noise * eps) / np.sqrt(P_FEAT) + 3.0) ** DEG + 1.0
    return X, C


def regrets(model, X, C):
    rs = []
    for x, c in zip(X, C):
        chat = model(torch.as_tensor(x, dtype=torch.float32)).detach().numpy()
        chat = np.maximum(chat, 1e-6)
        z_pred, z_true = shortest_path(chat), shortest_path(c)
        opt = c @ z_true
        rs.append((c @ z_pred - opt) / max(abs(opt), 1e-9))
    return float(np.mean(rs))


def main():
    rng = np.random.default_rng(SEED)
    B = rng.binomial(1, 0.5, (N_EDGES, P_FEAT)).astype(float)
    Xtr, Ctr = gen_data(1000, B, rng)
    Xte, Cte = gen_data(1000, B, rng)

    def train(loss_type, epochs=250, lr=None):
        torch.manual_seed(SEED)
        model = nn.Linear(P_FEAT, N_EDGES)
        if lr is None:
            # SPO 损失量级 ~O(10^3)（40 边 × 成本 ~80），Adam 需小 lr；
            # 且 SPO 梯度（2(z_s - z_p)，元素 ∈ {-2,0,2}）方差大，须更多 epoch 收敛
            lr = 5e-3 if loss_type == "mse" else 2e-4
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        Xt = torch.as_tensor(Xtr, dtype=torch.float32)
        Ct = torch.as_tensor(Ctr, dtype=torch.float32)
        bs = 32
        for ep in range(epochs):
            perm = torch.randperm(len(Xt))
            tot = 0.0
            for i in range(0, len(Xt), bs):
                idx = perm[i:i + bs]
                xb, cb = Xt[idx], Ct[idx]
                ch = model(xb)
                if loss_type == "mse":
                    loss = ((ch - cb) ** 2).mean()
                else:  # SPO+（Elmachtoub & Grigas 2022 Def. 3）
                    # L = (2ĉ-c)ᵀz*(c) − (2ĉ-c)ᵀz*(2ĉ−c)；梯度 2(z*(c) − z*(2ĉ−c))
                    # 2026-09-15 修正：旧实现 (2ĉ-c)ᵀz_s − ĉᵀz*(ĉ) 梯度=2z_s−z*(ĉ)，
                    # 完美预测时梯度 = z*(c) ≠ 0（会把模型推离真解）——符号与参照解双错
                    losses = []
                    for j in range(len(xb)):
                        p = ch[j].detach().numpy()
                        p = np.maximum(p, 1e-6)
                        c_true_j = Ctr[idx.numpy()[j]]
                        z_true = torch.as_tensor(shortest_path(c_true_j), dtype=torch.float32)
                        z_s = torch.as_tensor(
                            shortest_path(np.maximum(2 * p - c_true_j, 1e-6)),
                            dtype=torch.float32)
                        losses.append(torch.dot(2 * ch[j] - cb[j], z_true)
                                      - torch.dot(2 * ch[j] - cb[j], z_s))
                    loss = torch.stack(losses).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach())
            if (ep + 1) % 50 == 0:
                print(f"  [{loss_type}] epoch {ep+1}/{epochs} loss={tot/(len(Xt)//bs):.1f}",
                      flush=True)
        return model

    for lt in ("mse", "spo"):
        m = train(lt)
        r_tr = regrets(m, Xtr, Ctr)
        r_te = regrets(m, Xte, Cte)
        print(f"{lt.upper():5s}  train regret {r_tr:.5f}   test regret {r_te:.5f}", flush=True)
    print("\n判定：若 SPO test regret < MSE test regret ⇒ 实现通过正向对照。")


if __name__ == "__main__":
    main()

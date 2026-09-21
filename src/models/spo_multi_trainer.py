"""B3a：多物料 LIRP 的 SPO+ cost 模式训练器（端到端决策聚焦，论文 2 主方法）。

设定：
- 不确定性 = 弧运输成本乘数 c_ij（~LogNormal，特征 = 有噪观测）
- 预测器 = 共享 MLP（log 观测 -> 乘数），MSE 与 SPO+ 两种损失
- SPO+（Elmachtoub & Grigas）：L = (2ĉ-c)ᵀx̂(2ĉ-c) - ĉᵀx̂(ĉ)，
  梯度 dL/dĉ_ij = 2·xspo_ij - xpred_ij（对弧上所有 m,t 流量聚合）
  x̂ 由 LP 松懈求得（Mandi et al. 2020：松弛保持决策对齐）
- 评估：两阶段协议（ŷ(ĉ) MILP -> 真成本追索 LP，遗憾 vs y*(c)）
"""
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.cdw_lirp_multi import CDW_LIRP_MultiModel, MATERIALS  # noqa: E402


class CostMLP(nn.Module):
    def __init__(self, n_feat=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, x):  # x: [B, A, n_feat] -> [B, A]
        return self.net(x).squeeze(-1)


class CostScenario:
    """一个场景：固定拓扑 + 每弧真值乘数与有噪观测特征。

    成本敏感版（v2）：全设施兼容（选址由运输经济性驱动）、大流量、
    紧容量（0.45×总量，必须精选 2-3 个设施）、开厂成本收窄、乘数方差大。
    """

    def __init__(self, seed, n_sites=5, n_fac=3, n_dem=6, periods=2, grid=50.0,
                 distort=False):
        rng = np.random.default_rng(seed)
        S = list(range(1, n_sites + 1)); F = list(range(10, 10 + n_fac))
        Dp = list(range(20, 20 + n_dem)); T = list(range(periods))
        coords = {n: (rng.uniform(0, grid), rng.uniform(0, grid)) for n in S + F + Dp}
        nt = {**{s: "site" for s in S}, **{f: "facility" for f in F}, **{d: "demand" for d in Dp}}
        theta = np.stack([rng.dirichlet(np.ones(len(MATERIALS))) for _ in S])
        D_sites = rng.uniform(80, 160, n_sites)
        supply = {(s, m, t): float(D_sites[i] * theta[i, k] / periods)
                  for i, s in enumerate(S) for k, m in enumerate(MATERIALS) for t in T}
        demand = {(d, m, t): float(rng.uniform(0.5, 2.5) * D_sites.mean() / n_dem / periods)
                  for d in Dp for m in MATERIALS for t in T}
        compat = {(f, m): 1.0 for f in F for m in MATERIALS}  # 全兼容
        dist = {(i, j): float(np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]))
                for i in S + F for j in F + Dp if i != j}
        # 弧列表（成本乘数作用的弧）
        self.arcs = list(dist.keys())
        self.dist = dist
        # 真值乘数与特征（K 个有噪观测，对数域噪声 sigma）
        self.sigma = 0.35
        self.c_true = np.array([float(np.exp(rng.normal(0, 0.7))) for _ in self.arcs])
        K = 4
        if distort:
            # 非可逆特征：观察 (log c)^2 的带噪版本——符号不可辨，MSE 回归到双支均值而偏倚
            base = np.log(self.c_true) ** 2
            self.feats = np.stack([base + rng.normal(0, self.sigma, len(self.arcs))
                                   for _ in range(K)], axis=1)
        else:
            self.feats = np.stack([np.log(self.c_true) + rng.normal(0, self.sigma, len(self.arcs))
                                   for _ in range(K)], axis=1)  # [A, K]
        self.kwargs = dict(coords=coords, node_types=nt, supply=supply, demand=demand,
                           capacity={f: float(0.45 * D_sites.sum()) for f in F},
                           opening_cost={f: float(rng.uniform(4000, 4600)) for f in F},
                           holding_cost={f: 2.0 for f in F}, compat=compat,
                           proc_cost={m: 5.0 for m in MATERIALS},
                           revenue={"mineral": 12.0, "wood": 15.0, "gypsum": 6.0,
                                    "plastic_mix": 10.0, "general": 2.0},
                           vehicle_capacity=100.0, penalty_cost=10000.0,
                           time_periods=T)
        self.S, self.T = S, T

    def tc(self, mult: np.ndarray) -> Dict:
        return {a: self.dist[a] * mult[k] for k, a in enumerate(self.arcs)}

    def arc_flow(self, sol: Dict) -> np.ndarray:
        """解 -> 每弧聚合流量 [A]（材料/周期求和）。"""
        f = np.zeros(len(self.arcs))
        pos = {a: k for k, a in enumerate(self.arcs)}
        for (i, j, m, t), v in sol["x"].items():
            f[pos[(i, j)]] += v
        return f


def solve_flows(sc: CostScenario, mult: np.ndarray, lp: bool, y_fixed=None,
                time_limit: float = 20.0):
    mdl = CDW_LIRP_MultiModel(transport_cost=sc.tc(mult), lp_relaxation=lp,
                              y_fixed=y_fixed, **sc.kwargs)
    sol = mdl.solve(time_limit=time_limit)
    return sol, mdl


def spo_loss_and_grad(sc: CostScenario, c_pred: torch.Tensor):
    """SPO+ 损失（Elmachtoub & Grigas 2022 Def. 3；torch 可微：x̂ 以常数张量进入）。

    L = (2ĉ−c)ᵀx*(c) − (2ĉ−c)ᵀx*(2ĉ−c)；梯度 2(x*(c) − x*(2ĉ−c))。
    2026-09-15 修正：旧实现 (2ĉ−c)ᵀxs − ĉᵀx*(ĉ) 在完美预测 ĉ=c 时梯度 = x*(c) ≠ 0
    （会把模型推离真解），符号与参照解双错——此为 v1-v4 全负结果的可疑根因。"""
    with torch.no_grad():
        p = c_pred.detach().cpu().numpy()
        if getattr(sc, "_xt", None) is None:  # x*(c) 每场景固定，只解一次并缓存
            sol_t, _ = solve_flows(sc, sc.c_true, lp=True)
            if sol_t is None:
                return None
            sc._xt = torch.as_tensor(sc.arc_flow(sol_t), dtype=torch.float32)
        sol_s, _ = solve_flows(sc, 2 * p - sc.c_true, lp=True)
        if sol_s is None:
            return None
        xs = torch.as_tensor(sc.arc_flow(sol_s), dtype=torch.float32)
    ct = torch.as_tensor(sc.c_true, dtype=torch.float32)
    loss = torch.dot(2 * c_pred - ct, sc._xt) - torch.dot(2 * c_pred - ct, xs)
    return loss


def train(sc_train, loss_type="spo", epochs=25, lr=None, seed=0):
    # SPO+ 损失以成本计（千级），对 O(1) 乘数参数的梯度远大于 MSE——
    # 须归一化（除以场景最优值的量级）并使用小学习率，否则 Adam 步长爆炸
    if lr is None:
        lr = 5e-3 if loss_type == "mse" else 5e-4
    torch.manual_seed(seed)
    net = CostMLP()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X = torch.as_tensor(np.array([sc.feats for sc in sc_train]), dtype=torch.float32)
    Y = torch.as_tensor(np.array([sc.c_true for sc in sc_train]), dtype=torch.float32)
    with torch.no_grad():  # 归一化尺度：训练集真值成本目标量级
        zs = [solve_flows(sc, sc.c_true, lp=True)[0] for sc in sc_train[:5]]
        scale = float(np.mean([abs(s["objective"]) for s in zs if s])) if any(zs) else 1e3
    for ep in range(epochs):
        for b in range(len(sc_train)):
            opt.zero_grad()
            pred = net(X[b:b + 1])[0]
            if loss_type == "mse":
                loss = ((pred - Y[b]) ** 2).mean()
            else:
                loss = spo_loss_and_grad(sc_train[b], pred)
                if loss is None:
                    continue
                loss = loss / scale  # 量纲归一
            loss.backward()
            opt.step()
        if (ep + 1) % 5 == 0:
            print(f"  {loss_type} epoch {ep+1}: loss={float(loss):.4f}", flush=True)
    return net


@torch.no_grad()
def eval_cost(sc_test, net=None, mode="mse"):
    regs = []
    for sc in sc_test:
        sol_true, _ = solve_flows(sc, sc.c_true, lp=False)
        if sol_true is None:
            continue
        if net is None:
            mult = np.ones(len(sc.arcs))
        else:
            mult = net(torch.as_tensor(sc.feats, dtype=torch.float32)).numpy()
        y_hat, _ = solve_flows(sc, mult, lp=False)
        if y_hat is None:
            continue
        yf = {f: v for f, v in y_hat["y"].items()}
        rec, _ = solve_flows(sc, sc.c_true, lp=True, y_fixed=yf)
        regs.append(rec["objective"] - sol_true["objective"])
    return float(np.nanmean(regs)), float(np.nanmedian(regs))

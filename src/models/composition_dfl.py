"""组成驱动多物料 LIRP 的 DFL 训练器（E3 首个实验）。

设定（承接论文1 demand 模式经验 + plan2 视觉主链路）：
- 每个场景 = 一个 LIRP 实例（拓扑、兼容矩阵、成本从 site_scenario_bridge 采样）
- 每个产废工地 s 的真值：物料別供给 s_m = D_s * theta_{s,m}（来自组成管线）
- "视觉特征"代理：N 张有噪快照 m_j = (s_m/N) * LogNormal(0, sigma_obs)
  ——模拟感知层输出（掩码面积×先验的观测噪声），后续换冻结视觉嵌入即可
- 预测器：MLP(快照特征) -> 物料別供给 ŝ_m（softplus）
- 损失：MSE / Pinball(τ)（τ 网格由验证集遗憾选定——决策感知分位数）
- 评估（两阶段协议，遗憾恒 ≥ 0）：
    regret = cost( ŷ(ŝ), recourse 重优化 @ 真值 ) - z*(真值)
- 基线：no-vision（训练集边际均值组成）、oracle（真 θ/D）

经验依据（论文1）：RHS 不确定性下 SPO+ 不适用（需目标系数线性进入），
FY 扰动在离散第一阶段决策上零梯度，Pinball 内点分位数（0.90-0.95）最优。
"""
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.cdw_lirp_multi import CDW_LIRP_MultiModel, MATERIALS  # noqa: E402


# ---------------------------------------------------------------------------
# 场景与特征
# ---------------------------------------------------------------------------

def snapshot_features(supply_vec: np.ndarray, n_snap: int, sigma: float,
                      rng: np.random.Generator) -> np.ndarray:
    """真值物料別供给 -> N 张有噪快照矩阵 [N, M]（log 观测噪声）。"""
    per = supply_vec / n_snap
    snaps = per[None, :] * np.exp(rng.normal(0, sigma, size=(n_snap, len(supply_vec))))
    return np.log1p(snaps)  # log 稳定特征


class SitePredictor(nn.Module):
    """快照特征 [N, n_in] -> 供给估计 [n_out]（均值池化 + MLP）。"""

    def __init__(self, n_in: int, n_out: int = None, hidden: int = 64):
        super().__init__()
        n_out = n_out or n_in
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out), nn.Softplus(),
        )

    def forward(self, snaps: torch.Tensor) -> torch.Tensor:
        return self.net(snaps).mean(dim=-2)  # 对快照维求均值


# ---------------------------------------------------------------------------
# 圳策评估（两阶段协议）
# ---------------------------------------------------------------------------

def solve_y(model_kwargs: Dict, supply_by_material: Dict, lp: bool = False,
            time_limit: float = 30.0):
    """给定供给构建多物料 LIRP 并解出 y。supply_by_material: {(s,m,t): val}"""
    mdl = CDW_LIRP_MultiModel(lp_relaxation=lp, supply=supply_by_material, **model_kwargs)
    sol = mdl.solve(time_limit=time_limit)
    if sol is None:
        return None, None
    return {f: sol["y"][f] for f in mdl.facilities}, sol["objective"]


def two_stage_regret(model_kwargs: Dict, pred_supply: Dict, true_supply: Dict) -> float:
    """遗憾 = cost(ŷ(pred), 追索@true) - z*(true)。"""
    y_hat, _ = solve_y(model_kwargs, pred_supply)                      # 第一阶段
    mdl_true = CDW_LIRP_MultiModel(supply=true_supply, **model_kwargs)
    sol_star = mdl_true.solve(time_limit=30.0)
    if y_hat is None or sol_star is None:
        return float("nan")
    mdl_rec = CDW_LIRP_MultiModel(supply=true_supply, y_fixed=y_hat,
                                  lp_relaxation=True, **model_kwargs)  # 追索 LP
    sol_rec = mdl_rec.solve(time_limit=30.0)
    if sol_rec is None:
        return float("nan")
    return float(sol_rec["objective"] - sol_star["objective"])


def supply_dict(theta_sites: np.ndarray, D_sites: np.ndarray,
                S: List[int], T: List[int]) -> Dict:
    """[S,M] 组成 + [S] 总量 -> {(s,m,t): 供给}。"""
    out = {}
    for si, s in enumerate(S):
        for t in T:
            for mi, mm in enumerate(MATERIALS):
                out[(s, mm, t)] = float(D_sites[si] * theta_sites[si, mi] / len(T))
    return out


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_predictor(train_feats, train_targets, loss_type="mse", tau=0.9,
                    epochs=300, lr=1e-3, seed=0) -> SitePredictor:
    torch.manual_seed(seed)
    model = SitePredictor(train_feats.shape[-1], train_targets.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.FloatTensor(train_feats)                 # [B, N, M]
    Y = torch.FloatTensor(train_targets)               # [B, M]
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X)
        if loss_type == "mse":
            loss = ((pred - Y) ** 2).mean()
        elif loss_type == "hybrid":
            # 总量 pinball（分位数决策价值）+ 配比 MSE（保持单纯形结构不扭曲）
            tot_p, tot_y = pred.sum(dim=-1), Y.sum(dim=-1)
            d = tot_y - tot_p
            loss_pb = torch.maximum(tau * d, (tau - 1) * d).mean()
            mix_p = pred / (tot_p[:, None] + 1e-9)
            mix_y = Y / (tot_y[:, None] + 1e-9)
            loss = loss_pb + 10.0 * ((mix_p - mix_y) ** 2).mean()
        else:  # pinball（逐维）
            d = Y - pred
            loss = torch.maximum(tau * d, (tau - 1) * d).mean()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def predict_supply(model, feats) -> np.ndarray:
    return model(torch.FloatTensor(feats)).numpy()

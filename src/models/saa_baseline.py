"""SAA（样本平均逼近）extensive-form 基线——组成模式下与视觉/无视觉方法同台对比。

形式：min_y (1/K) Σ_k [ 追索成本(y, s_k) ] + 开设成本，s_k 为从训练分布采样的
K 个组成/总量实现（每工地按原型簇采样）。决策只有第一阶段 y 进入第二阶段各场景。

变量规模 ~K×850（K=5 时约 4,250），超 pip 社区版限制 -> 走 cplex.exe Studio 通路。
返回 y_SAA，进入与其它方法相同的两阶段评估协议。
"""
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.cdw_lirp_multi import MATERIALS  # noqa: E402
from src.models.cplex_studio_solver import solve_via_studio  # noqa: E402

try:
    from docplex.mp.model import Model
except ImportError:
    Model = None


class _Shim:
    """让 solve_via_studio 接受裸 docplex 模型（无 .y 回填，取 raw）。"""

    def __init__(self, m):
        self.model = m


def solve_saa_y(kwargs: Dict, draws: List[Dict], time_limit: float = 60.0,
                risk: str = "mean", beta: float = 0.8) -> Dict:
    """kwargs: make_scenario 的拓扑参数（不含 supply）；draws: K 个 supply 字典。

    risk="mean"：期望目标 (1/K)Σ 追索（经典 SAA / SAA-pred）。
    risk="cvar"：Rockafellar–Uryashev CVaR_β 目标（A2 风险厌恶版）：
        min  Σ o_f y_f + η + (1/((1-β)K)) Σ_k u_k,  u_k ≥ 追索_k − η, u_k ≥ 0。
        对一阶段不可逆选址，尾部场景（组成翻转×容量紧张）由 η/u 显式定价。
    """
    if Model is None:
        raise ImportError("docplex required")
    S, F = kwargs["coords"].keys(), None
    node_types = kwargs["node_types"]
    S = [n for n, t in node_types.items() if t == "site"]
    F = [n for n, t in node_types.items() if t == "facility"]
    Dp = [n for n, t in node_types.items() if t == "demand"]
    T = kwargs["time_periods"]
    M = MATERIALS
    K = len(draws)
    dist = kwargs.get("transport_cost")
    if dist is None:
        dist = {}
        for i in S + F:
            for j in F + Dp:
                if i != j:
                    dist[(i, j)] = float(np.hypot(kwargs["coords"][i][0] - kwargs["coords"][j][0],
                                                  kwargs["coords"][i][1] - kwargs["coords"][j][1]))
    m = Model(name="SAA_extensive")
    y = m.binary_var_dict(F, name="y")
    big_m = (len(S) + len(F) - 1) * len(M) * kwargs["vehicle_capacity"]
    recourse_exprs = []  # 未加权的追索线性表达式 r_k
    for k, sup in enumerate(draws):
        x = {}
        for i in S + F:
            for j in F + Dp:
                if i == j:
                    continue
                for mm in M:
                    for t in T:
                        x[i, j, mm, t] = m.continuous_var(lb=0, name=f"x{k}_{i}_{j}_{mm}_{t}")
        I = m.continuous_var_dict([(f, mm, t) for f in F for mm in M for t in T], lb=0,
                                  name=f"I{k}")
        sd = m.continuous_var_dict([(d, mm, t) for d in Dp for mm in M for t in T], lb=0,
                                   name=f"sd{k}")
        ss = m.continuous_var_dict([(s, mm, t) for s in S for mm in M for t in T], lb=0,
                                   name=f"ss{k}")
        r_k = (m.sum(dist[(i, j)] * x[i, j, mm, t] for (i, j, mm, t) in x)
               + m.sum(kwargs["holding_cost"][f] * I[f, mm, t] for f in F for mm in M for t in T)
               + m.sum(kwargs["proc_cost"][mm] * x[i, f, mm, t]
                       for i in S + F for f in F for mm in M for t in T
                       if (i, f, mm, t) in x and i != f)
               - m.sum(kwargs["revenue"][mm] * x[f, d, mm, t]
                       for f in F for d in Dp for mm in M for t in T if (f, d, mm, t) in x)
               + kwargs["penalty_cost"] * (m.sum(sd[d, mm, t] for d in Dp for mm in M for t in T)
                                           + m.sum(ss[s, mm, t] for s in S for mm in M for t in T)))
        recourse_exprs.append(r_k)
        for f in F:
            for mm in M:
                for t in T:
                    infl = m.sum(x[i, f, mm, t] for i in S + F if (i, f, mm, t) in x)
                    outf = m.sum(x[f, j, mm, t] for j in F + Dp if (f, j, mm, t) in x)
                    if t == T[0]:
                        m.add_constraint(I[f, mm, t] == infl - outf)
                    else:
                        m.add_constraint(I[f, mm, t] == I[f, mm, T[T.index(t) - 1]] + infl - outf)
            for t in T:
                m.add_constraint(m.sum(I[f, mm, t] for mm in M) <= kwargs["capacity"][f] * y[f])
                infl_tot = m.sum(x[i, f, mm, t] for i in S + F for mm in M if (i, f, mm, t) in x)
                m.add_constraint(infl_tot <= big_m * y[f])
        for f in F:
            for mm in M:
                if not kwargs["compat"].get((f, mm), 0):
                    for t in T:
                        for i in S + F:
                            if (i, f, mm, t) in x:
                                m.add_constraint(x[i, f, mm, t] == 0)
        for d in Dp:
            for mm in M:
                for t in T:
                    m.add_constraint(m.sum(x[f, d, mm, t] for f in F if (f, d, mm, t) in x)
                                     + sd[d, mm, t] == kwargs["demand"].get((d, mm, t), 0.0))
        for s in S:
            for mm in M:
                for t in T:
                    m.add_constraint(m.sum(x[s, f, mm, t] for f in F if (s, f, mm, t) in x)
                                     + ss[s, mm, t] == sup.get((s, mm, t), 0.0))
    opening = m.sum(kwargs["opening_cost"][f] * y[f] for f in F)
    if risk == "mean":
        m.minimize(opening + (1.0 / K) * sum(recourse_exprs))
    else:  # CVaR_β（Rockafellar–Uryashev）
        eta = m.continuous_var(name="eta")
        u = m.continuous_var_list(K, lb=0, name="u")
        for k in range(K):
            m.add_constraint(u[k] >= recourse_exprs[k] - eta)
        m.minimize(opening + eta + (1.0 / ((1 - beta) * K)) * m.sum(u))
    sol = solve_via_studio(_Shim(m), time_limit=time_limit)
    if sol is None:
        return None
    return {f: sol["raw"].get(f"y_{f}", 0.0) for f in F}

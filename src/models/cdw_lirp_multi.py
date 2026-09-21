"""多物料（multi-commodity）CDW-LIRP MILP —— plan2 像素到决策方案的决策层。

相对单物料版（src/models/cdw_lirp.py，论文1已验证）的扩展：
1. 物料集合 M：CODD 10 类聚合为可配置物料组（默认 5 组）
2. 供给由视觉组成驱动：supply[(s,m,t)] = theta[s,m] * D[s,t]（组成×总量）
3. 设施-物料兼容矩阵 compat[f,m]：设施只能处理特定物料
   —— 组成错估 => 开错设施类型的决策机制显式化
4. 物料别处理成本 proc_cost[m] 与再生收益 revenue[m]（负成本进目标）
5. 库存/流量/需求全部增加物料维 I[f,m,t], x[i,j,m,t], demand[(d,m,t)]

接口与单物料版对齐（y_fixed / lp_relaxation / solve / evaluate），便于 DFL 训练循环复用。
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from docplex.mp.model import Model
    DOCPLEX_AVAILABLE = True
except ImportError:
    DOCPLEX_AVAILABLE = None
    Model = None

# CODD 10 类 -> 物料组聚合（可配置）
MATERIAL_GROUPS = {
    "mineral": ["concrete", "brick", "tile", "stone"],   # 矿物类（再生骨料）
    "wood": ["wood"],
    "gypsum": ["gypsum_board", "foam"],
    "plastic_mix": ["plastic", "pipes"],
    "general": ["general_w"],
}
MATERIALS = list(MATERIAL_GROUPS.keys())


class CDW_LIRP_MultiModel:
    """Multi-material multi-period LIRP (docplex/CPLEX)."""

    def __init__(
        self,
        coords: Dict[int, Tuple[float, float]],
        node_types: Dict[int, str],
        supply: Dict[Tuple[int, str, int], float],   # (site, material, t) -> 供给量
        demand: Dict[Tuple[int, str, int], float],    # (demand_pt, material, t) -> 再生产品需求
        capacity: Dict[int, float],
        opening_cost: Dict[int, float],
        holding_cost: Dict[int, float],
        compat: Dict[Tuple[int, str], float],         # (facility, material) -> 1/0 兼容
        proc_cost: Dict[str, float],                  # material -> 单位处理成本
        revenue: Dict[str, float],                    # material -> 单位再生收益
        transport_cost: Optional[Dict[Tuple[int, int], float]] = None,
        vehicle_capacity: float = 100.0,
        penalty_cost: float = 10000.0,
        time_periods: Optional[List[int]] = None,
        lp_relaxation: bool = False,
        y_fixed: Optional[Dict[int, float]] = None,
        materials: Optional[List[str]] = None,
    ):
        if not DOCPLEX_AVAILABLE:
            raise ImportError("docplex required: pip install docplex")
        self.M = materials or MATERIALS
        self.coords = coords
        self.node_types = node_types
        self.supply = supply
        self.demand = demand
        self.capacity = capacity
        self.opening_cost = opening_cost
        self.holding_cost = holding_cost
        self.compat = compat
        self.proc_cost = proc_cost
        self.revenue = revenue
        self.vehicle_capacity = vehicle_capacity
        self.penalty_cost = penalty_cost
        self.lp_relaxation = lp_relaxation
        self.y_fixed = y_fixed

        self.sites = [n for n, t in node_types.items() if t == "site"]
        self.facilities = [n for n, t in node_types.items() if t == "facility"]
        self.demand_points = [n for n, t in node_types.items() if t == "demand"]
        self.nodes = list(coords.keys())
        self.periods = time_periods or sorted({t for (_, _, t) in supply.keys()} |
                                              {t for (_, _, t) in demand.keys()})

        if transport_cost is None:
            self.transport_cost = {
                (i, j): float(np.hypot(coords[i][0] - coords[j][0],
                                       coords[i][1] - coords[j][1]))
                for i in self.nodes for j in self.nodes if i != j
            }
        else:
            self.transport_cost = transport_cost

        self.model = None
        self._build_model()

    def _build_model(self):
        m = Model(name="CDW_LIRP_Multi")
        self.model = m
        S, F, Dp, T, M = self.sites, self.facilities, self.demand_points, self.periods, self.M

        y = (m.continuous_var_dict(F, lb=0, ub=1, name="y") if self.lp_relaxation
             else m.binary_var_dict(F, name="y"))
        if self.y_fixed is not None:
            for f in F:
                if f in self.y_fixed:
                    m.add_constraint(y[f] == float(self.y_fixed[f]), ctname=f"fix_y_{f}")

        # 物料别流量：site/facility -> facility/demand
        x = {}
        for i in S + F:
            for j in F + Dp:
                if i == j:
                    continue
                for mm in M:
                    for t in T:
                        x[i, j, mm, t] = m.continuous_var(lb=0, name=f"x_{i}_{j}_{mm}_{t}")

        I = m.continuous_var_dict([(f, mm, t) for f in F for mm in M for t in T],
                                  lb=0, name="I")
        s_demand = m.continuous_var_dict([(d, mm, t) for d in Dp for mm in M for t in T],
                                         lb=0, name="s_dem")
        s_supply = m.continuous_var_dict([(s, mm, t) for s in S for mm in M for t in T],
                                         lb=0, name="s_sup")

        obj = m.sum(self.opening_cost[f] * y[f] for f in F)
        obj += m.sum(self.transport_cost[(i, j)] * x[i, j, mm, t]
                     for (i, j, mm, t) in x)
        obj += m.sum(self.holding_cost[f] * I[f, mm, t] for f in F for mm in M for t in T)
        obj += m.sum(self.proc_cost[mm] * x[i, f, mm, t]
                     for i in S + F for f in F for mm in M for t in T if (i, f, mm, t) in x and i != f)
        obj -= m.sum(self.revenue[mm] * x[f, d, mm, t]
                     for f in F for d in Dp for mm in M for t in T if (f, d, mm, t) in x)
        obj += self.penalty_cost * (m.sum(s_demand[d, mm, t] for d in Dp for mm in M for t in T)
                                    + m.sum(s_supply[s, mm, t] for s in S for mm in M for t in T))
        m.minimize(obj)

        # 1) 物料别设施库存平衡
        for f in F:
            for mm in M:
                for t in T:
                    inflow = m.sum(x[i, f, mm, t] for i in S + F if (i, f, mm, t) in x)
                    outflow = m.sum(x[f, j, mm, t] for j in F + Dp if (f, j, mm, t) in x)
                    if t == T[0]:
                        m.add_constraint(I[f, mm, t] == inflow - outflow,
                                         ctname=f"bal_{f}_{mm}_{t}")
                    else:
                        pt = T[T.index(t) - 1]
                        m.add_constraint(I[f, mm, t] == I[f, mm, pt] + inflow - outflow,
                                         ctname=f"bal_{f}_{mm}_{t}")

        # 2) 设施容量（物料共享）× 开设
        for f in F:
            for t in T:
                m.add_constraint(m.sum(I[f, mm, t] for mm in M) <= self.capacity[f] * y[f],
                                 ctname=f"cap_{f}_{t}")

        # 2b) 处理吞吐量激活：设施有流入必须开设（堵住"过境零库存"漏洞）
        big_m = (len(S) + len(F) - 1) * len(M) * self.vehicle_capacity
        for f in F:
            for t in T:
                inflow_total = m.sum(x[i, f, mm, t] for i in S + F
                                     for mm in M if (i, f, mm, t) in x)
                m.add_constraint(inflow_total <= big_m * y[f],
                                 ctname=f"act_{f}_{t}")

        # 3) 设施-物料兼容：不兼容则禁止流入
        for f in F:
            for mm in M:
                if not self.compat.get((f, mm), 0):
                    for t in T:
                        for i in S + F:
                            if (i, f, mm, t) in x:
                                m.add_constraint(x[i, f, mm, t] == 0,
                                                 ctname=f"nc_{i}_{f}_{mm}_{t}")

        # 4) 物料别需求满足
        for d in Dp:
            for mm in M:
                for t in T:
                    inflow = m.sum(x[f, d, mm, t] for f in F if (f, d, mm, t) in x)
                    m.add_constraint(inflow + s_demand[d, mm, t]
                                     == self.demand.get((d, mm, t), 0.0),
                                     ctname=f"dem_{d}_{mm}_{t}")

        # 5) 物料别供给收集
        for s in S:
            for mm in M:
                for t in T:
                    outflow = m.sum(x[s, f, mm, t] for f in F if (s, f, mm, t) in x)
                    m.add_constraint(outflow + s_supply[s, mm, t]
                                     == self.supply.get((s, mm, t), 0.0),
                                     ctname=f"sup_{s}_{mm}_{t}")

        # 6) 弧容量
        for (i, j, mm, t) in x:
            m.add_constraint(x[i, j, mm, t] <= self.vehicle_capacity,
                             ctname=f"vc_{i}_{j}_{mm}_{t}")

        self.y, self.x, self.I = y, x, I
        self.s_demand, self.s_supply = s_demand, s_supply

    def solve(self, log_output: bool = False, time_limit: Optional[float] = None) -> Optional[Dict]:
        if time_limit is not None:
            self.model.set_time_limit(time_limit)
        sol = self.model.solve(log_output=log_output)
        if sol is None:
            return None
        return {
            "objective": sol.objective_value,
            "y": {f: sol.get_value(self.y[f]) for f in self.facilities},
            "x": {k: v for k, v in
                  ((k, sol.get_value(var)) for k, var in self.x.items()) if v > 1e-6},
            "I": {k: sol.get_value(var) for k, var in self.I.items()},
            "s_demand": {k: sol.get_value(var) for k, var in self.s_demand.items()},
            "s_supply": {k: sol.get_value(var) for k, var in self.s_supply.items()},
        }

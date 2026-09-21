"""组成估计 -> 多物料 LIRP 实例 的场景桥（E3 脚手架）。

把 vision_composition / composition_estimator 输出的每图组成 CSV：
1. 聚合成 K 个伪"产废工地"（每工地 = N 张图的 θ/D 分布）
2. 映射 CODD 10 类 -> 5 物料组（与 cdw_lirp_multi.MATERIAL_GROUPS 一致）
3. 生成 supply/demand/compat，构建多物料 LIRP 并用 CPLEX 求解最优 z*
4. 输出 DFL 训练元组：(工地特征 x, 组成 θ, 总量 D, 最优决策 y*, 最优目标 z*)

用法:
    python -m src.vision.site_scenario_bridge --comp-csv <composition_testing_trial.csv> \
        --n-sites 5 --n-samples 60 --out <scenarios.json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.cdw_lirp_multi import CDW_LIRP_MultiModel, MATERIAL_GROUPS, MATERIALS  # noqa: E402
from src.vision.codd_to_coco import CATS  # noqa: E402

# CODD 类 -> 物料组索引
CLASS2GROUP = {c: g for g, cs in MATERIAL_GROUPS.items() for c in cs}


def load_thetas(csv_path: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = [f"theta_med_{c}" for c in CATS]
    thetas = df[cols].to_numpy()             # [n_img, 10]
    thetas = thetas / thetas.sum(axis=1, keepdims=True)
    D = df["D_med"].to_numpy()
    return thetas, D


def to_group_space(theta10: np.ndarray) -> np.ndarray:
    """[..., 10] CODD 类组成 -> [..., 5] 物料组组成。"""
    g = np.zeros(theta10.shape[:-1] + (len(MATERIALS),))
    for j, c in enumerate(CATS):
        gi = MATERIALS.index(CLASS2GROUP[c])
        g[..., gi] += theta10[..., j]
    return g / g.sum(axis=-1, keepdims=True)


def build_lirp(theta_sites: np.ndarray, D_sites: np.ndarray, rng,
               n_facilities=3, n_demand=6, periods=2, grid=50.0):
    S = list(range(1, len(D_sites) + 1))
    F = list(range(10, 10 + n_facilities))
    Dp = list(range(20, 20 + n_demand))
    coords = {n: (rng.uniform(0, grid), rng.uniform(0, grid)) for n in S + F + Dp}
    node_types = {**{s: "site" for s in S}, **{f: "facility" for f in F},
                  **{d: "demand" for d in Dp}}
    supply, demand = {}, {}
    for si, s in enumerate(S):
        for t in range(periods):
            for mi, mm in enumerate(MATERIALS):
                supply[(s, mm, t)] = D_sites[si] * theta_sites[si, mi] / periods
    for d in Dp:
        for t in range(periods):
            for mm in MATERIALS:
                demand[(d, mm, t)] = float(rng.uniform(0.5, 2.5) * D_sites.mean() / n_demand / periods)
    # 兼容矩阵：每个设施 2-3 个物料组（含 mineral 的概率高——矿物主导流）
    compat = {}
    for f in F:
        groups = {"mineral"}
        while len(groups) < rng.integers(2, 4):
            groups.add(str(MATERIALS[rng.integers(0, len(MATERIALS))]))
        for mm in MATERIALS:
            compat[(f, mm)] = 1.0 if mm in groups else 0.0
    model = CDW_LIRP_MultiModel(
        coords=coords, node_types=node_types, supply=supply, demand=demand,
        capacity={f: float(1.2 * D_sites.sum() / max(len(F), 1)) for f in F},
        opening_cost={f: float(rng.uniform(4000, 6000)) for f in F},
        holding_cost={f: 2.0 for f in F}, compat=compat,
        proc_cost={mm: 5.0 for mm in MATERIALS},
        revenue={"mineral": 12.0, "wood": 15.0, "gypsum": 6.0,
                 "plastic_mix": 10.0, "general": 2.0},
        vehicle_capacity=100.0, penalty_cost=10000.0,
        time_periods=list(range(periods)),
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-csv", required=True)
    ap.add_argument("--n-sites", type=int, default=5)
    ap.add_argument("--imgs-per-site", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-scale", type=float, default=2000.0,
                    help="图像快照质量->工地日产量标定系数（日帧数假设，E5 校准对象）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    thetas, D = load_thetas(args.comp_csv)
    thetas_g = to_group_space(thetas)
    n_img = len(thetas_g)

    records = []
    for k in range(args.n_samples):
        # 每个样本 = 每工地随机抽 imgs_per_site 张图组成
        site_theta, site_D = [], []
        for s in range(args.n_sites):
            idx = rng.choice(n_img, size=args.imgs_per_site, replace=True)
            w = D[idx] / D[idx].sum()
            site_theta.append((w[:, None] * thetas_g[idx]).sum(axis=0))  # 质量加权组成
            site_D.append(D[idx].sum())
        theta_sites = np.stack(site_theta)
        D_sites = np.array(site_D) * args.d_scale
        model = build_lirp(theta_sites, D_sites, rng)
        sol = model.solve(time_limit=30)
        if sol is None:
            continue
        records.append({
            "theta": theta_sites.round(4).tolist(),
            "D": D_sites.round(3).tolist(),
            "y_star": {str(f): int(v + 0.5) for f, v in sol["y"].items() if v > 0.5},
            "z_star": round(sol["objective"], 2),
        })
        if (k + 1) % 10 == 0:
            print(f"{k+1}/{args.n_samples} solved", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(out, "w"), indent=1)
    opens = [sum(r["y_star"].values()) for r in records]
    print(f"{len(records)} scenarios -> {out}")
    print(f"facilities opened per scenario: mean={np.mean(opens):.2f}, "
          f"min={min(opens)}, max={max(opens)}")


if __name__ == "__main__":
    main()

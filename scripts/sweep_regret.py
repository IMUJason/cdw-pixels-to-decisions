"""B2：遗憾细化——簇纯度 / 每工地图像数 敏感性 + 分场景类型遗憾。

基于 run_composition_dfl_vision 的原型设定，扫描 purity∈{0.5,0.7,0.9} ×
imgs∈{4,8,16}（vision+MSE vs no-vision，30 场景/配置），并在主配置输出
逐场景遗憾与"组成敏感场景"标记（oracle y* 与 no-vision y* 是否不同）。

用法: python scripts/sweep_regret.py
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

from src.models.cdw_lirp_multi import MATERIALS  # noqa: E402
from src.models.composition_dfl import train_predictor, predict_supply, supply_dict  # noqa: E402
from run_composition_dfl import make_scenario  # noqa: E402
from run_composition_dfl_vision import load_split  # noqa: E402
from src.models.cdw_lirp_multi import CDW_LIRP_MultiModel  # noqa: E402
from run_composition_dfl import solve_y  # noqa: E402


def site_indices(pool_n, clusters, n_imgs, purity, rng):
    idx, cl_id = [], rng.integers(0, 4)
    cl = clusters[cl_id]
    n_in = int(purity * n_imgs)
    idx = np.concatenate([rng.choice(cl, n_in, replace=True),
                          rng.choice(pool_n, n_imgs - n_in, replace=True)])
    return idx, cl_id


def run_config(P_tr, G_tr, LAB_tr, P_te, G_te, LAB_te, models, mean_mix, mean_tot,
               n_sites=5, n_imgs=8, purity=0.7, n_sc=30, seed=0):
    rng = np.random.default_rng(seed)
    cl_tr = [np.where(LAB_tr == c)[0] for c in range(4)]
    cl_te = [np.where(LAB_te == c)[0] for c in range(4)]
    regs_v, regs_n, sensitive = [], [], []
    for k in range(n_sc):
        theta, D, feats, arch = [], [], [], []
        for s in range(n_sites):
            idx, cid = site_indices(len(P_te), cl_te, n_imgs, purity, rng)
            gm = G_te[idx].sum(0) * 2000.0
            D.append(gm.sum()); theta.append(gm / max(gm.sum(), 1e-9))
            A = P_te[idx]
            feats.append(A / (A.sum(1, keepdims=True) + 1e-9))
            arch.append(cid)
        theta = np.stack(theta); D = np.array(D)
        kwargs, S, T, _, _ = make_scenario(rng, np.eye(5), np.ones(1),
                                           n_sites=n_sites, sites_override=(theta, D))
        true_sup = supply_dict(theta, D, S, T)
        _, z_star = solve_y(kwargs, true_sup)
        # 组成敏感性：no-vision 最优 y 是否等于 oracle y*
        th_n = np.tile(mean_mix, (n_sites, 1))
        sup_n = supply_dict(th_n, np.full(n_sites, mean_tot / n_sites), S, T)
        y_n, _ = solve_y(kwargs, sup_n)
        y_o, _ = solve_y(kwargs, true_sup)
        sens = any(abs(y_n[f] - y_o[f]) > .5 for f in y_n)
        sensitive.append(sens)
        # vision
        pred = predict_supply(models['mse'], np.array(feats))
        pred = np.clip(pred, 1e-6, None)
        sup_v = supply_dict(pred / pred.sum(1, keepdims=True), pred.sum(1), S, T)
        for sup, reg_list in ((sup_v, regs_v), (sup_n, regs_n)):
            y_hat, _ = solve_y(kwargs, sup)
            rec = CDW_LIRP_MultiModel(supply=true_sup, y_fixed=y_hat,
                                      lp_relaxation=True, **kwargs).solve(time_limit=30)
            reg_list.append(rec["objective"] - z_star if rec else np.nan)
    rv, rn = np.nanmean(regs_v), np.nanmean(regs_n)
    sens_rate = float(np.mean(sensitive))
    # 敏感子集上的遗憾
    sens_idx = [i for i, s in enumerate(sensitive) if s]
    rv_s = np.nanmean([regs_v[i] for i in sens_idx]) if sens_idx else 0.0
    rn_s = np.nanmean([regs_n[i] for i in sens_idx]) if sens_idx else 0.0
    return {"purity": purity, "n_imgs": n_imgs, "vision": rv, "novision": rn,
            "gain": 1 - rv / rn if rn else 0, "sens_rate": sens_rate,
            "vision_sens": rv_s, "novision_sens": rn_s}


def main():
    P_tr, G_tr, _, LAB_tr = load_split("train")
    P_te, G_te, _, LAB_te = load_split("test")
    # 训练（原型采样，purity 0.7, imgs 8 —— 与主实验一致）
    rng = np.random.default_rng(1)
    cl_tr = [np.where(LAB_tr == c)[0] for c in range(4)]
    xs, ys = [], []
    for _ in range(500):
        idx, _ = site_indices(len(P_tr), cl_tr, 8, 0.7, rng)
        xs.append(P_tr[idx] / (P_tr[idx].sum(1, keepdims=True) + 1e-9))
        ys.append(G_tr[idx].sum(0) * 2000.0)
    model = train_predictor(np.array(xs), np.array(ys), "mse", epochs=400, seed=0)
    mean_mix = np.array(ys).mean(0); mean_mix = mean_mix / mean_mix.sum()
    mean_tot = float(np.array(ys).sum(1).mean())

    results = []
    for purity in (0.5, 0.7, 0.9):
        for n_imgs in (4, 8, 16):
            r = run_config(P_tr, G_tr, LAB_tr, P_te, G_te, LAB_te,
                           {"mse": model}, mean_mix, mean_tot,
                           n_imgs=n_imgs, purity=purity, n_sc=30, seed=42)
            results.append(r)
            print(f"purity={r['purity']} imgs={r['n_imgs']:2d} | "
                  f"vision={r['vision']:9.1f} no-vision={r['novision']:9.1f} "
                  f"gain={r['gain']:+.1%} | sens_rate={r['sens_rate']:.0%} "
                  f"(敏感子集 vision={r['vision_sens']:.0f} vs no={r['novision_sens']:.0f})", flush=True)
    out = Path(r"D:\2026 cdw 0911 datasets\processed\sweep_regret.json")
    json.dump(results, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()

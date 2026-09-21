"""E3 视觉特征版：s@640 预测面积（真实感知噪声）-> 组成/总量 -> 多物料 LIRP。

与 run_composition_dfl.py（对数正态模拟噪声）的区别：
- 特征 x = 该工地 N 张图的**预测**逐类面积 [N,10]（真实感知误差，来自 E1 模型）
- 真值 = 同一批图的**真值**多边形面积 × 面密度中点 × d-scale（伪真值协议）
- 预测器需自学 px->质量标定 + 感知噪声修正 —— 论文"冻结视觉编码器 + 决策头"的雏形

用法:
    python scripts/run_composition_dfl_vision.py --n-train 100 --n-val 30 --n-test 40
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

from src.models.cdw_lirp_multi import MATERIALS  # noqa: E402
from src.models.composition_dfl import train_predictor, predict_supply, supply_dict  # noqa: E402
from src.vision.material_priors import CODD_PRIORS, CODD_METERS_PER_PIXEL  # noqa: E402
from src.vision.codd_to_coco import CATS  # noqa: E402
from src.vision.site_scenario_bridge import CLASS2GROUP  # noqa: E402
from run_composition_dfl import make_scenario, eval_method  # noqa: E402

FEAT_DIR = Path(r"D:\2026 cdw 0911 datasets\processed\area_features")

# 面密度中点（t/m²）：密度中点 × 厚度中点
AD_MID = {c: ((CODD_PRIORS[c].density_lo + CODD_PRIORS[c].density_hi) / 2)
          * ((CODD_PRIORS[c].thick_lo + CODD_PRIORS[c].thick_hi) / 2) for c in CATS}
M2PP2 = CODD_METERS_PER_PIXEL ** 2


def load_split(split: str):
    pred = pd.read_csv(FEAT_DIR / f"pred_areas_{split}.csv").set_index("image")
    gt = pd.read_csv(FEAT_DIR / f"gt_areas_{split}.csv").set_index("image")
    common = pred.index.intersection(gt.index)
    P = pred.loc[common, CATS].to_numpy() * M2PP2      # 预测 px² -> m²
    G = gt.loc[common, CATS].to_numpy()                # 真值 m²（已标定）
    # 真值质量 [n, 5 物料组]
    G_mass = np.zeros((len(common), len(MATERIALS)))
    for j, c in enumerate(CATS):
        G_mass[:, MATERIALS.index(CLASS2GROUP[c])] += G[:, j] * AD_MID[c]
    from sklearn.cluster import KMeans
    lab = KMeans(4, n_init=10, random_state=0).fit_predict(G_mass / (G_mass.sum(1, keepdims=True) + 1e-9))
    return P, G_mass, list(common), lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--n-sites", type=int, default=5)
    ap.add_argument("--d-scale", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=r"D:\2026 cdw 0911 datasets\processed\dfl_e3_vision.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    P_tr, G_tr, _, LAB_tr = load_split("train")
    P_te, G_te, _, LAB_te = load_split("test")

    def sample_xy(P, G, n_scenarios, seed, pool_lab):
        r = np.random.default_rng(seed)
        xs, ys = [], []
        clusters = [np.where(pool_lab == c)[0] for c in range(4)]
        for _ in range(n_scenarios):
            for s in range(args.n_sites):
                cl = clusters[r.integers(0, 4)]
                n_in = int(0.7 * args.n_imgs)
                idx = np.concatenate([r.choice(cl, n_in, replace=True),
                                      r.choice(len(P), args.n_imgs - n_in, replace=True)])
                xs.append(P[idx] / (P[idx].sum(1, keepdims=True) + 1e-9))  # [N,10] 逐图面积份额
                ys.append(G[idx].sum(axis=0) * args.d_scale)
        return np.array(xs), np.array(ys)

    Xtr, Ytr = sample_xy(P_tr, G_tr, args.n_train, args.seed + 1, LAB_tr)
    print(f"train sites: {len(Xtr)}  feature: {Xtr.shape[1:]}")

    models = {"mse": train_predictor(Xtr, Ytr, "mse", epochs=400, seed=args.seed)}
    models["hybrid_0.9"] = train_predictor(Xtr, Ytr, "hybrid", tau=0.9, epochs=400, seed=args.seed)

    # 场景：先从测试图像池采样工地真值，再按其规模生成需求/容量（保证量纲一致）
    r2 = np.random.default_rng(args.seed + 2)
    sc2, sc_site_idx = [], []
    for _ in range(args.n_val + args.n_test):
        clusters_te = [np.where(LAB_te == c)[0] for c in range(4)]
        theta_sites, D_sites, site_idx = [], [], []
        for _ in range(args.n_sites):
            cl = clusters_te[r2.integers(0, 4)]
            n_in = int(0.7 * args.n_imgs)
            idx = np.concatenate([r2.choice(cl, n_in, replace=True),
                                  r2.choice(len(P_te), args.n_imgs - n_in, replace=True)])
            gm = G_te[idx].sum(axis=0) * args.d_scale
            D_sites.append(gm.sum())
            theta_sites.append(gm / max(gm.sum(), 1e-9))
            site_idx.append(idx)
        sc2.append(make_scenario(rng, np.eye(len(MATERIALS)), np.ones(1),
                                 n_sites=args.n_sites,
                                 sites_override=(np.stack(theta_sites), np.array(D_sites))))
        sc_site_idx.append(site_idx)
    val_sc, test_sc = sc2[:args.n_val], sc2[args.n_val:]

    def feats_from_idx(site_idx, P_pool):
        A = np.array([P_pool[idx] for idx in site_idx])
        return A / (A.sum(2, keepdims=True) + 1e-9)

    def method_pred(model_or_none, kind, P_pool, idx_list):
        def f(k, kwargs, S, T, theta, D):
            if kind == "oracle":
                return supply_dict(theta, D, S, T)
            if kind == "novision":
                mix = Ytr.mean(axis=0); mix = mix / mix.sum()
                tot = float(Ytr.sum(axis=1).mean()) * len(S)
                return supply_dict(np.tile(mix, (len(S), 1)), np.full(len(S), tot / len(S)), S, T)
            pred = predict_supply(model_or_none, feats_from_idx(idx_list[k], P_pool))
            pred = np.clip(pred, 1e-6, None)
            return supply_dict(pred / pred.sum(axis=1, keepdims=True),
                               pred.sum(axis=1), S, T)
        return f

    val_rows = []
    for name in ("mse", "hybrid_0.9"):
        row = eval_method(name, val_sc, method_pred(models[name], "model", P_te, sc_site_idx))
        val_rows.append(row); print("VAL", row)
    best = min(val_rows, key=lambda r: r["mean_regret"])["method"]

    rows = [eval_method("no-vision(mean mix)", test_sc, method_pred(None, "novision", P_te, sc_site_idx)),
            eval_method("vision+MSE", test_sc, method_pred(models["mse"], "model", P_te, sc_site_idx)),
            eval_method("vision+" + best, test_sc, method_pred(models[best], "model", P_te, sc_site_idx)),
            eval_method("oracle(true comp)", test_sc, method_pred(None, "oracle", P_te, sc_site_idx))]
    out = {"config": vars(args), "val": val_rows, "test": rows}
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("\n=== TEST regret (vision features, two-stage) ===")
    print(f"{'method':24s} {'mean':>12s} {'median':>12s} {'worst':>12s}")
    for r in rows:
        print(f"{r['method']:24s} {r['mean_regret']:12.1f} {r['median']:12.1f} {r['worst']:12.1f}")
    print("->", args.out)


if __name__ == "__main__":
    main()

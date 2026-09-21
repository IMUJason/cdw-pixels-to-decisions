"""d-scale 敏感性：帧→日产量标定常数在 {500,1000,2000,4000} 下的方法排序稳定性。

d_scale 是场景生成器里唯一无法从数据标定的自由常数（belt 帧质量 → 工地日产量）。
它不仅线性缩放供给，还改变问题的经济结构（开设成本固定，供给越小开设越不划算），
因此方法排序是否随 d_scale 变化必须实测而非理论假设。

用法: python scripts/run_dscale_sensitivity.py --n-test 20
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

import run_baselines_compare as rbc  # noqa: E402
from src.models.cdw_lirp_multi import CDW_LIRP_MultiModel  # noqa: E402
from src.models.composition_dfl import train_predictor, predict_supply, supply_dict  # noqa: E402
from src.models.saa_baseline import solve_saa_y  # noqa: E402
from run_composition_dfl import make_scenario, solve_y  # noqa: E402
from run_composition_dfl_vision import load_split  # noqa: E402


def run_one(dscale, n_test, seed, model, mean_mix, mean_tot, P_te, G_te, cl_te,
            P_tr, G_tr, cl_tr):
    rbc.D_SCALE = dscale  # sample_site 运行时读取模块全局
    rng = np.random.default_rng(seed)
    rng_saa = np.random.default_rng(777)
    methods = ["no-vision", "vision", "SAA(train)", "oracle"]
    regs = {m: [] for m in methods}
    for k in range(n_test):
        theta, D, feats = [], [], []
        for s in range(rbc.N_SITES):
            th, d, feat = rbc.sample_site(P_te, G_te, cl_te, rng)
            theta.append(th); D.append(d); feats.append(feat)
        theta = np.stack(theta); D = np.array(D)
        kwargs, S, T, _, _ = make_scenario(rng, np.eye(5), np.ones(1), n_sites=rbc.N_SITES,
                                           cap_factor=0.6, sites_override=(theta, D))
        true_sup = supply_dict(theta, D, S, T)
        y_o, z_star = solve_y(kwargs, true_sup)

        def regret(y_hat):
            rec = CDW_LIRP_MultiModel(supply=true_sup, y_fixed=y_hat,
                                      lp_relaxation=True, **kwargs).solve(time_limit=30)
            return rec["objective"] - z_star if rec else np.nan

        regs["oracle"].append(regret(y_o))
        th_n = np.tile(mean_mix, (rbc.N_SITES, 1))
        y_n, _ = solve_y(kwargs, supply_dict(
            th_n, np.full(rbc.N_SITES, mean_tot / rbc.N_SITES), S, T))
        regs["no-vision"].append(regret(y_n))
        pred = np.clip(predict_supply(model, np.array(feats)), 1e-6, None)
        th_hat = pred / pred.sum(1, keepdims=True)
        y_v, _ = solve_y(kwargs, supply_dict(th_hat, pred.sum(1), S, T))
        regs["vision"].append(regret(y_v))
        draws = []
        for _ in range(rbc.K_SAA):
            th_k, D_k = [], []
            for s in range(rbc.N_SITES):
                t2, d2, _ = rbc.sample_site(P_tr, G_tr, cl_tr, rng_saa)
                th_k.append(t2); D_k.append(d2)
            draws.append(supply_dict(np.stack(th_k), np.array(D_k), S, T))
        y_s = solve_saa_y(kwargs, draws, time_limit=30)
        regs["SAA(train)"].append(
            regret({f: 1.0 if v > .5 else 0.0 for f, v in y_s.items()}) if y_s else np.nan)
    return {m: float(np.nanmean(v)) for m, v in regs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dscales", nargs="+", type=float,
                    default=[500, 1000, 2000, 4000])
    args = ap.parse_args()

    P_tr, G_tr, _, LAB_tr = load_split("train")
    P_te, G_te, _, LAB_te = load_split("test")
    cl_tr = [np.where(LAB_tr == c)[0] for c in range(4)]
    cl_te = [np.where(LAB_te == c)[0] for c in range(4)]

    results = {}
    for ds in args.dscales:
        rbc.D_SCALE = ds  # 训练目标尺度随 ds 变化（部署方认知一致）
        rng0 = np.random.default_rng(1)
        xs, ys = [], []
        for _ in range(500):
            th, D, feat = rbc.sample_site(P_tr, G_tr, cl_tr, rng0)
            xs.append(feat); ys.append(D * th)
        xs, ys = np.array(xs), np.array(ys)
        model = train_predictor(xs, ys, "mse", epochs=400, seed=0)
        mean_mix = ys.mean(0) / ys.mean(0).sum()
        mean_tot = float(ys.sum(1).mean())
        summ = run_one(ds, args.n_test, args.seed, model, mean_mix, mean_tot,
                       P_te, G_te, cl_te, P_tr, G_tr, cl_tr)
        results[str(ds)] = summ
        print(f"d_scale={ds:6.0f}  " + "  ".join(f"{m}={v:.0f}" for m, v in summ.items()),
              flush=True)

    out = Path(r"D:\2026 cdw 0911 datasets\processed\dscale_sensitivity.json")
    json.dump({"config": vars(args), "results": results}, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()

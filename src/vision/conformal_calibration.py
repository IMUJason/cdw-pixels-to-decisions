"""E2b: split conformal 校准 —— 组成向量 θ 的区间收紧。

问题：先验 MC 区间在名义 90% 下实际覆盖 96-97%（过宽）。
方法：split conformal——在校准集上取每类残差 |θ_true - θ_pred| 的经验分位数
q_c，测试区间 = [θ_pred - q_c, θ_pred + q_c]（Marginal 保覆盖；per-class 校准）。
报告：校准前后覆盖率/平均区间宽度对比。

用法:
    python -m src.vision.conformal_calibration --pred-csv <pred.csv> --gt-csv <gt.csv> \
        --alpha 0.1 --out <calibrated.csv>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.vision.codd_to_coco import CATS  # noqa: E402


def conformal_theta(pred_csv: str, gt_csv: str, alpha: float = 0.1,
                    cal_frac: float = 0.5, seed: int = 0, nonzero_only: bool = False):
    """nonzero_only=True：分位数只从"真值非零"的组分残差中取（条件校准），
    避免大量零组分的平凡零残差把 q 压小导致非零组分上严重失覆盖。"""
    pred = pd.read_csv(pred_csv)
    gt = pd.read_csv(gt_csv)
    m = pred.merge(gt, on="image", suffixes=("_p", "_g"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(m))
    n_cal = int(len(m) * cal_frac)
    cal, tst = m.iloc[idx[:n_cal]], m.iloc[idx[n_cal:]]

    out_rows = []
    q = {}
    for j, c in enumerate(CATS):
        r = np.abs(cal[f"theta_med_{c}_p"].to_numpy() - cal[f"theta_med_{c}_g"].to_numpy())
        if nonzero_only:
            mask = cal[f"theta_med_{c}_g"].to_numpy() > 1e-6
            r = r[mask] if mask.sum() >= 10 else r  # 非零样本过少则退回边际
        n = len(r)
        k = min(int(np.ceil((n + 1) * (1 - alpha))) - 1, n - 1)
        q[c] = float(np.sort(r)[max(k, 0)])

    # 测试集：conformal 区间 vs 先验 MC 区间
    stats = {"coverage_mc": [], "coverage_cf": [], "width_mc": [], "width_cf": []}
    for _, row in tst.iterrows():
        for c in CATS:
            g = row[f"theta_med_{c}_g"]
            p = row[f"theta_med_{c}_p"]
            lo_mc, hi_mc = row[f"theta_p05_{c}_p"], row[f"theta_p95_{c}_p"]
            lo_cf, hi_cf = max(0.0, p - q[c]), min(1.0, p + q[c])
            stats["coverage_mc"].append(lo_mc <= g <= hi_mc)
            stats["coverage_cf"].append(lo_cf <= g <= hi_cf)
            stats["width_mc"].append(hi_mc - lo_mc)
            stats["width_cf"].append(hi_cf - lo_cf)
            out_rows.append({"image": row["image"], "class": c,
                             "theta_pred": p, "theta_true": g,
                             "lo_conformal": lo_cf, "hi_conformal": hi_cf})
    s = {k: (float(np.mean(v)) if "coverage" in k else float(np.mean(v)))
         for k, v in stats.items()}
    return pd.DataFrame(out_rows), s, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nonzero-only", action="store_true",
                    help="仅在真值非零组分上校准（条件校准）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df, s, q = conformal_theta(args.pred_csv, args.gt_csv, args.alpha, args.cal_frac,
                               args.seed, nonzero_only=args.nonzero_only)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"nominal coverage = {1 - args.alpha:.0%}")
    print(f"prior-MC intervals : coverage={s['coverage_mc']:.3f}, mean width={s['width_mc']:.4f}")
    print(f"conformal intervals: coverage={s['coverage_cf']:.3f}, mean width={s['width_cf']:.4f}")
    print(f"width reduction: {1 - s['width_cf'] / s['width_mc']:.1%}")
    print("per-class q:", {c: round(v, 4) for c, v in q.items()})
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

"""B3a 实验：SPO+ vs MSE vs no-prediction（多物料 LIRP 弧成本模式）。

用法: python scripts/run_spo_multi.py --n-train 40 --n-test 30
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.spo_multi_trainer import CostScenario, train, eval_cost  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=r"D:\2026 cdw 0911 datasets\processed\spo_multi.json")
    ap.add_argument("--distort", action="store_true", help="非可逆特征（MSE 偏倚场景）")
    args = ap.parse_args()

    sc_train = [CostScenario(1000 + i, distort=args.distort) for i in range(args.n_train)]
    sc_test = [CostScenario(2000 + i, distort=args.distort) for i in range(args.n_test)]

    print("training SPO+ ...", flush=True)
    net_spo = train(sc_train, "spo", epochs=args.epochs)
    print("training MSE ...", flush=True)
    net_mse = train(sc_train, "mse", epochs=args.epochs)

    rows = {}
    rows["no-pred(unit mult)"] = eval_cost(sc_test, None)
    rows["MSE"] = eval_cost(sc_test, net_mse)
    rows["SPO+"] = eval_cost(sc_test, net_spo)
    print("\n=== TEST regret (cost mode, two-stage) ===")
    print(f"{'method':22s} {'mean':>12s} {'median':>12s}")
    for k, (m, md) in rows.items():
        print(f"{k:22s} {m:12.1f} {md:12.1f}")
    json.dump({k: list(v) for k, v in rows.items()},
              open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()

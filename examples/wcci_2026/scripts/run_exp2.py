#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path

import fuzzylinguistics as fl


def call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**supported)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp2_music_genre_taste.json")
    ap.add_argument("--results", default="results/exp2")
    ap.add_argument("--truth_threshold", type=float, default=0.65)
    ap.add_argument("--w_focus", type=float, default=0.35)
    ap.add_argument("--w_complexity", type=float, default=0.15)
    ap.add_argument("--plot_mfs", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    fls = fl.setup_fls(args.config)
    fls.w_focus = args.w_focus
    fls.w_complexity = args.w_complexity

    Path(args.results).mkdir(parents=True, exist_ok=True)

    call_with_supported_kwargs(
        fls.generate_fls_one_model,
        results_dir=args.results,
        truth_threshold=args.truth_threshold,
        simplify_fls=True,
        plot_membership_functions=args.plot_mfs,

        save_initial_fls_txt=True,
        save_initial_fls_csv=True,
        save_initial_fls_latex=True,

        save_simplified_fls_txt=True,
        save_simplified_fls_csv=True,
        save_simplified_fls_latex=True,

        save_simplification_graph_gexf=True,
    )

    print("Elapsed_s=", round(time.time() - t0, 3))


if __name__ == "__main__":
    main()

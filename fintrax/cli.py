import argparse
import importlib

# backtest isn't in `all` on purpose, run it separately
STAGES = ["scrape", "parse", "label", "train", "score", "signals"]
EXTRA = ["backtest"]
MODULES = {"label": "dataset"}


def main() -> None:
    ap = argparse.ArgumentParser(prog="fintrax")
    ap.add_argument("stages", nargs="+", choices=STAGES + EXTRA + ["all"])
    ap.add_argument("--tickers", nargs="*", help="scrape only these tickers")
    args = ap.parse_args()

    from fintrax.config import load_config

    cfg = load_config()
    stages = STAGES if "all" in args.stages else args.stages
    for stage in stages:
        print("==", stage, "==")
        mod = importlib.import_module(f"fintrax.{MODULES.get(stage, stage)}")
        if stage == "scrape":
            mod.run(cfg, only=args.tickers)
        else:
            mod.run(cfg)


if __name__ == "__main__":
    main()

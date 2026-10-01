"""Command-line interface."""
import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(prog="engramforge", description="Conditional-memory sparse LLM framework (Engram).")
    p.add_argument("--config", default=None, help="Path to .yaml or .json config")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Self-check: backends, memory module, MoE, benchmarks registered")
    sub.add_parser("train", help="Train the trainable mock backend (skill -> 1.0)")
    sub.add_parser("bench", help="Run the knowledge/reasoning/code/math benchmark suite")
    sub.add_parser("scaling", help="Fit the U-shaped sparsity-allocation scaling law")
    sub.add_parser("offload", help="Benchmark host-memory embedding offload")
    sub.add_parser("analyze", help="Mechanistic analysis: which layers Engram relieves")
    sub.add_parser("serve", help="Start the OpenAI-compatible stdlib HTTP server")
    sub.add_parser("demo", help="Run the end-to-end demo script")
    return p


def main(argv=None):
    from .config import load_config

    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "doctor":
        from .doctor import run
        run(cfg)
    elif args.command == "train":
        from .train.trainer import run_train
        run_train(cfg)
    elif args.command == "bench":
        from .bench.run import run_all
        run_all(cfg)
    elif args.command == "scaling":
        from .train.scaling import run_scaling
        run_scaling(cfg)
    elif args.command == "offload":
        from .bench.offload import run_offload
        run_offload(cfg)
    elif args.command == "analyze":
        from .bench.analysis import run_analysis
        run_analysis(cfg)
    elif args.command == "serve":
        from .serving.server import run_server
        run_server(cfg)
    elif args.command == "demo":
        from .demo import run
        run(cfg)
    else:
        build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

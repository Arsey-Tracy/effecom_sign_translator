import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config


def setup_logging():
    log_level = config.get("logging.level", "INFO")
    log_format = config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get("logging.file", "logs/usl_translator.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="USL Translator - Ugandan Sign Language Live Translator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.run collect    # Collect training data
  python -m src.run train      # Train the model
  python -m src.run infer      # Run live inference
        """,
    )

    parser.add_argument(
        "mode",
        choices=["collect", "train", "infer"],
        help="Operation mode: collect data, train model, or run inference",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config.yaml in project root)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Override device for training/inference (default: from config)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs (default: from config)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override confidence threshold for inference (default: from config)",
    )

    args = parser.parse_args()

    if args.config:
        config.load(args.config)

    if args.device:
        config._config["training"]["device"] = args.device

    if args.epochs:
        config._config["training"]["epochs"] = args.epochs

    if args.confidence:
        config._config["inference"]["confidence_threshold"] = args.confidence

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Starting USL Translator in '{args.mode}' mode")

    try:
        if args.mode == "collect":
            from builder import collect_data
            collect_data()
        elif args.mode == "train":
            from train import train_model
            train_model()
        elif args.mode == "infer":
            from main import run_inference
            run_inference()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
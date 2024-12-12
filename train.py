import argparse

from source.config import ExperimentConfig
from source.experiment import Experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default=None, help="Path to config file"
    )
    parser.add_argument(
        "--save_config",
        type=str,
        default=None,
        help="Path to save the current config (JSON or YAML)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    args = parser.parse_args()

    # Load or initialize the config
    cfg = ExperimentConfig()
    if args.config_file:
        file_overrides = ExperimentConfig.from_file(args.config_file)
        cfg.update_from_dict(file_overrides)

    # Apply CLI overrides
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
        print(f"Overriding batch size to {args.batch_size}")

    # Save the current config if specified
    if args.save_config:
        cfg.save_to_file(args.save_config)
        print(f"Configuration saved to {args.save_config}")

    # Create and run the experiment
    exp = Experiment(cfg)
    exp.setup()
    exp.run()


if __name__ == "__main__":
    main()

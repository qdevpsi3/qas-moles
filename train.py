import argparse

from source.config import ExperimentConfig
from source.experiment import Experiment


def main():
    # Here we handle argparse if we want to specify config file or other overrides
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default=None, help="Path to config file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.config_file:
        file_overrides = ExperimentConfig.from_file(args.config_file)
        cfg.update_from_dict(file_overrides)

    cfg.save_to_file("configs/experiment_config.yaml")
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size

    exp = Experiment(cfg)
    exp.run()


if __name__ == "__main__":
    main()

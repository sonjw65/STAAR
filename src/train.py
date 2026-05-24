import argparse
import importlib
import json
import os

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig


DATA_ROOT_PATH = "./datasets"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        required=True,
    )
    parser.add_argument(
        "--gpus",
        default="0",
    )
    return parser.parse_args()


def get_config_class(model_name):
    model_module = importlib.import_module(f"basicts.models.{model_name}")
    return getattr(model_module, f"{model_name}Config")


def get_model_name(model_config):
    model_module = model_config["module"]
    return model_module.split("basicts.models.", 1)[1].split(".", 1)[0]


def load_default_config(default_config, config):
    loaded_config = dict(default_config)
    for key, value in config.items():
        if key == "default":
            continue
        if isinstance(value, dict) and isinstance(loaded_config.get(key), dict):
            loaded_config[key] = load_default_config(loaded_config[key], value)
        else:
            loaded_config[key] = value
    return loaded_config


def load_config(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    default_path = config_dict.get("default")
    if default_path is None:
        return config_dict

    if not os.path.isabs(default_path):
        default_path = os.path.join(os.path.dirname(json_file_path), default_path)

    default_config = load_config(default_path)
    return load_default_config(default_config, config_dict)


def build_training_config(json_file_path):
    config_dict = load_config(json_file_path)
    model_name = get_model_name(config_dict["model"])
    config_class = get_config_class(model_name)

    config_dict["callbacks"] = [
        callback
        for callback in config_dict.get("callbacks", [])
        if callback.get("name") != "WandbMeterLogger"
    ]

    for key, value in config_dict.items():
        if key != "model_config":
            config_dict[key] = BasicTSForecastingConfig._construct_obj(value)

    config_dict["model_config"] = config_class(**config_dict["model_config"])

    return BasicTSForecastingConfig(**config_dict)


def get_ckpt_dir(cfg):
    return os.path.join("checkpoints", cfg.model.__name__, cfg.dataset_name)


def setup_config(args):
    os.environ.setdefault("BASICTS_DATA_ROOT", DATA_ROOT_PATH)

    cfg = build_training_config(args.cfg)

    cfg.gpus = args.gpus
    cfg.gpu_num = len(args.gpus.split(",")) if args.gpus else 0

    cfg.ckpt_save_dir = get_ckpt_dir(cfg)

    return cfg


def main():
    cfg = setup_config(parse_args())
    BasicTSLauncher.launch_training(cfg)


if __name__ == "__main__":
    main()

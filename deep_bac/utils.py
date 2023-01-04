import argparse

from deep_bac.modelling.data_types import DeepBacConfig


def get_config(args: argparse.Namespace) -> DeepBacConfig:
    return DeepBacConfig()

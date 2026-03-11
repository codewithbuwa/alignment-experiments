import json
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train_val_split(n: int, val_fraction: float, seed: int):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0,1)")
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    val_size = int(round(n * val_fraction))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return train_idx, val_idx


def update_latest_paths(label: str, path: str, latest_path_file: str = "results/_latest_paths.txt"):
    ensure_dir(os.path.dirname(latest_path_file))
    entries = {}
    if os.path.exists(latest_path_file):
        with open(latest_path_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                key, val = line.split("\t", 1)
                entries[key] = val
    entries[label] = path
    with open(latest_path_file, "w", encoding="utf-8") as f:
        for key in sorted(entries.keys()):
            f.write(f"{key}\t{entries[key]}\n")


def export_report_figure(src_path: str, report_name: str, report_dir: str = "report/figures"):
    if not os.path.exists(src_path):
        return
    ensure_dir(report_dir)
    dst_path = os.path.join(report_dir, report_name)
    shutil.copyfile(src_path, dst_path)

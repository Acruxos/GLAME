import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *


class KGDataset(Dataset):
    def __init__(
        self, data_dir: str, ds_name: str, size: typing.Optional[int] = None, *args, **kwargs
    ):
        data_dir = Path(data_dir)

        if ds_name == "cf":
            data_loc = data_dir / "counterfact_graph.json"
        elif ds_name == "cf-one-hop":
            data_loc = data_dir / "cf_plus_graph.json"
            print("CF_PLUS_GRAPH")
        elif ds_name == "mquake":
            data_loc = data_dir / "mquake_graph.json"
        else:
            raise NotImplementedError(
                "can not found corresponding KG in /data/!")

        with open(data_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

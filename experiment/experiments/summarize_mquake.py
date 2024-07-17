import collections
from fuzzywuzzy import fuzz

# 已有答案和生成文本
import json
from pprint import pprint
from typing import List, Optional
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import numpy as np
from scipy.stats import hmean

from util.globals import *

stemmer = PorterStemmer()


def main(
    dir_name,
    runs: Optional[List],
    cases_lower_bound=None,
    cases_upper_bound=None,
    num_edits=None,
    num_hops=None,
    get_uncompressed=False,
    abs_path=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    generation_tot,generation_suc = 0,0
    single_hop_tot,single_hop_gen_suc, = 0,0
    instance_suc,generation_instance_suc = 0,0
    single_hop_acc = 0.0

    if num_hops is not None:
        if num_hops == 2:
            cases_upper_bound = 1001
        elif num_hops == 3:
            cases_lower_bound = 1000
            cases_upper_bound = 2001
        else:
            cases_lower_bound = 2000

    for run_dir in (RESULTS_DIR / dir_name if not abs_path else dir_name).iterdir():
        # Skip if we're not interested
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        files = list(run_dir.glob("*case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Could not decode {case_file} due to format error; skipping.")

            case_id = data["case_id"]
            if cases_lower_bound is not None and case_id <= cases_lower_bound:
                continue

            if cases_upper_bound is not None and case_id >= cases_upper_bound:
                break

            if num_edits is not None and data["portability"]["num_requests"] != num_edits:
                continue

            # 相似度阈值
            threshold = 70
            instance_flag = False
            generation_flag = False

            if "time" in data:
                cur_sum["time"].append(data["time"])

            for prefix in ["portability"]:
                port_text = data["portability"]["text"]
                port_target = [data["portability"]["portability_target"]
                               ]+data["portability"]["portability_target_alias"]

                for text in port_text:
                    for target in port_target:
                        similarity = fuzz.partial_ratio(text, target)

                        if similarity >= threshold:
                            generation_suc += 1
                            generation_flag = True
                            break

                    generation_tot = generation_tot+1

                # Probability metrics for which new should be lower (better) than true
                for key in ["question_prompts_probs"]:
                    for x in data[prefix]["multi_hop_result"][key]:
                        if x["target_true"] > x["target_new"]:
                            instance_flag = True
                            break

                    if prefix not in data or key not in data[prefix]["multi_hop_result"]:
                        continue

                    sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"

                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] > x["target_new"]
                                for x in data[prefix]["multi_hop_result"][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"]) -
                                np.exp(-x["target_true"])
                                for x in data[prefix]["multi_hop_result"][key]
                            ]
                        )
                    )

                # Accuracy-based evaluation metrics
                for key in ["question",]:
                    sum_key = f"{prefix}_{key}_acc"
                    key = f"{key}_prompts_correct"

                    if prefix not in data or key not in data[prefix]["multi_hop_result"]:
                        continue

                    cur_sum[sum_key].append(
                        np.mean(data[prefix]["multi_hop_result"][key]))

            
            if (instance_flag):
                instance_suc += 1

            if (generation_flag):
                generation_instance_suc += 1

        if len(cur_sum) == 0:
            continue

        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
            "port_generate_instance_acc": 0.0,
            "port_generate_acc": 0,
            "port_tot": 0,
            "port_suc": 0,
            "port_instance_acc": 0,
            "single_hop_gen_acc": 0.0,
            "single_hop_acc": single_hop_acc/num_items,
        }

        metadata["port_generate_instance_acc"] = generation_instance_suc/num_items
        metadata["port_generate_acc"] = generation_suc/generation_tot
        metadata["port_tot"] = generation_tot
        metadata["port_suc"] = generation_suc
        metadata["port_instance_acc"] = instance_suc/num_items

        uncompressed.append(dict(cur_sum, **metadata))

        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                # Constant multiplication scales linearly with mean and stddev
                cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)

        cur_sum.update(metadata)
        pprint(cur_sum)
        summaries.append(cur_sum)

    return uncompressed if get_uncompressed else summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--cases_lower_bound",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    parser.add_argument(
        "--cases_upper_bound",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=None,
        help="only summarize those cases whose [num_edits] is n"
    )
    parser.add_argument(
        "--num_hops",
        type=int,
        default=None,
        help="only summarize those cases whose [num_hops] is n"
    )
    args = parser.parse_args()

    main(
        args.dir_name,
        None if args.runs is None else args.runs.split(","),
        args.cases_lower_bound,
        args.cases_upper_bound,
        args.num_edits,
        args.num_hops,
    )

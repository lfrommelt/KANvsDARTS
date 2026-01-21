import json
import logging
import time
from types import SimpleNamespace
import numpy as np
import traceback
import threading

# import matplotlib.pyplot as plt
from src.utils import (
    PRIMITIVES,
    rmse,
    nrmse,
    sl_rmse,
    rmspe,
    r2,
    make_vectorized,
    NumericTensorJSONEncoder,
)
from pathlib import Path
import pickle
import sympy as sp
from itertools import permutations

import os
import sys

import torch

torch.set_default_dtype(torch.float64)
from sklearn.model_selection import ParameterGrid


from src.train_kan import train_kan
from src.utils import configure_logging

# print(train)
# import src.utils

log_level = logging.DEBUG
# log_level = logging.INFO
overwrite_log = True

dataset_folder = "datasets"
folder = "Results_KAN"
success_metric = nrmse
metric_norm = "std"
success_threshold = 0.2
# for debugging fast-forward training
scale_steps = 1.0 / 100  # /100
# baaaaaad
success_threshold = 0.18  # ??

success_threshold = (
    0.002  # last good recovery that still included exponential error was slightly above
)

success_threshold = 0.1  # for std-nrmse, slightly above was bad in k1 results, slightly below was minimally off

comment = "debug kan pipeline"

# List the variable names we want to record
keys = ("success_metric", "metric_norm", "success_threshold", "scale_steps", "comment")

# Grab their current values in *this* scope.
# If any name is missing, Python will raise a KeyError/NameError as requested.
vals = {key: locals()[key] for key in keys}

# Ensure the target directory exists
Path(f"{folder}").mkdir(exist_ok=True)

# Overwrite the file with a tidy, human-readable report
with open(f"{folder}/description.txt", "w", encoding="utf-8") as f:
    for key, v in vals.items():
        f.write(f'{key.replace("_", " ").title():20}: {v}\n')

# Import the W&B Python Library and log into W&B

import os


# random mapping index -> primitive
primitives = np.array(PRIMITIVES)
np.random.seed(0)
np.random.shuffle(primitives)

# mapping darts names -> kan names
NAME_TO_KEY = {
    "sin": "sin",
    "exp": "exp",
    "log": "log",
    "square": "x^2",
    "cube": "x^3",
    "recip": "1/x",
}


# first in alphabet -> varies slowest (evaluated last)2*2*2*3*3*2*3*3*2*2
grid = {
    "val_size": [20],  # [20, 40],
    "c_grid": [3, 5],
    "b_k": [3, 5],
    "structure_steps": [50, 100],  # ,200],
    "lamb_l1": [0.05, 0.01],  # , 0.002],
    "lamb_entropy": [0.00001, 0.000005],
    "loss_fn": [None],
    "a_lr": [1, 0.5],  # ,2],
    "symbolification_steps": [50, 100],  # ,200],
    "primitives": [list(NAME_TO_KEY.keys())],
    "finetune_steps": [100],  # [50, 100],
    "n_1d_fits": [3, 5],
}

# sizes=[1,2,3,4,5]

# conditions
c_multiplication = (False, True)
c_prior_knowledge = (True, False)
c_linear_transformations = (False, True)

# for simulating results for debugging
test_rng = np.random.default_rng(42)


def main():
    count = 0
    gridsize = len(ParameterGrid(grid))
    for config_kan in ParameterGrid(grid):

        count += 1
        print("hp config")
        print(*config_kan.items(), sep="\n")
        config_kan["structure_steps"] = max(
            1, int(config_kan["structure_steps"] * scale_steps)
        )
        config_kan["symbolification_steps"] = max(
            1, int(config_kan["symbolification_steps"] * scale_steps)
        )

        if count > 1:
            break
        for multiplication in c_multiplication:
            for prior_knowledge in c_prior_knowledge:
                """if prior_knowledge and (not multiplication) and (count == 2):
                continue"""
                for linear_transformations in c_linear_transformations:
                    # k=0 should be trivial
                    for k in range(1, 4):  # 5):
                        # all_failed=False
                        all_failed = True
                        for n_v in range(1, 4):
                            # mult only meaningful between at least two independent variables
                            if multiplication and (n_v == 1):
                                continue
                            for seed in range(3):
                                # Results_KAN/mult_False-prior_True-lin_True/k_1-nv_3-comb_1-seed_0.json
                                # for now we concider each seed an individual trial (iid problem!)

                                dataset_folder = Path(
                                    f"datasets/mult_{multiplication}-lin_{linear_transformations}"
                                )
                                with open(
                                    dataset_folder / f"equations_k{k}_nv{n_v}.txt", "rt"
                                ) as f:
                                    equations = [line.rstrip() for line in f]

                                for combination in range(5):
                                    # failed at
                                    # Results_KAN/mult_False-prior_False-lin_True/k_2-nv_2-comb_1-seed_0.json--------------------
                                    # Grid Condition 2 out of 256 (fuck...)
                                    # conntinued at
                                    # --------------------Results_KAN/mult_False-prior_False-lin_True/k_2-nv_2-comb_1-seed_0.json--------------------
                                    # Grid Condition 2 out of 25
                                    """if (count==1) and (k==1):
                                        continue
                                    if (count==1) and (k==2) and (n_v==1):
                                        continue"""
                                    # for continuing when interrupted, or replicating single runs
                                    """
                                    --------------------Results_KAN/mult_False-prior_True-lin_False/k_1-nv_2-comb_3-seed_0.json--------------------
                                    Grid Condition 1 out of 256
                                    """

                                    params = {
                                        "multiplication": False,
                                        "prior_knowledge": True,
                                        "linear_transformations": False,
                                        "k": 1,
                                        "n_v": 2,
                                        "combination": 3,
                                        "seed": 0,
                                        "count": 1,
                                    }
                                    local_vars = locals()

                                    # print([local_vars[key] for key in params.keys()])
                                    """if multiplication == True:
                                        continue
                                    if prior_knowledge == True:
                                        continue
                                    if linear_transformations == False:
                                        continue
                                    if n_v < 2:
                                        continue"""

                                    if all(
                                        list(
                                            local_vars[key] == val
                                            for key, val in params.items()
                                        )
                                    ):
                                        pass
                                    else:
                                        all_failed = False
                                        """print(
                                            {
                                                key: local_vars[key] == val
                                                for key, val in params.items()
                                            }
                                        )"""
                                        continue

                                    """if (count == 2) and all(
                                        list(
                                            local_vars[key] == val
                                            for key, val in params.items()
                                        )
                                    ):
                                        if k < 2:
                                            all_failed = False
                                            continue
                                        elif (k == 2) and (n_v < 2):
                                            all_failed = False
                                            continue
                                        elif (
                                            (k == 2) and (n_v == 2) and combination < 1
                                        ):
                                            all_failed = False
                                            continue"""
                                    # print(*list((str(key),local_vars[key]) for key, _ in params.items()),sep="\n")
                                    # count+=1
                                    """if not(linear_transformations) and not(multiplication):
                                        continue"""

                                    # start a trial
                                    path = f"{folder}/mult_{multiplication}-prior_{prior_knowledge}-lin_{linear_transformations}/k_{k}-nv_{n_v}-comb_{combination}-seed_{seed}.json"

                                    """if not (path=="Results_KAN/mult_False-prior_True-lin_True/k_2-nv_2-comb_1-seed_2.json"):
                                        continue"""

                                    """print(list((local_vars[key] , val) for key, val in params.items()))
                                    print(list(local_vars[key] == val for key, val in params.items()))
                                    """
                                    path = Path(path)
                                    run_logs = []
                                    ground_truth = equations[combination]

                                    # for k=0, all structures are the same
                                    if k == 0 and (combination > 0):
                                        continue
                                    for size in (1, 2, max(n_v, 3)):
                                        # here
                                        if size < 3:
                                            continue
                                        log_file = (
                                            Path(str(path)[:-5]) / f"size{size}.log"
                                        )
                                        configure_logging(
                                            log_file,
                                            level=log_level,
                                            overwrite=overwrite_log,
                                        )

                                        # config
                                        if size in (1, 2, 3):
                                            config = config_kan.copy()
                                        elif size in (4, 5):
                                            config = config_kan.copy()
                                        else:
                                            config = config_kan.copy()

                                        config["grid"] = config["c_grid"]
                                        config["k"] = config["b_k"]
                                        config["lr"] = config["a_lr"]
                                        config["seed"] = seed

                                        width = [n_v, size, 1]
                                        config["width"] = width
                                        config["dataset"] = (
                                            f"datasets_k{k}_nv{n_v}/{combination}.pkl"
                                        )
                                        config["combination"] = combination

                                        actual_primitives = config["primitives"].copy()
                                        if prior_knowledge:
                                            all_primitives = config["primitives"]
                                            actual_primitives = []
                                            for primitive in all_primitives:
                                                str_name = primitive
                                                if str_name == "square":
                                                    str_ = "**2"
                                                elif str_name == "cube":
                                                    str_ = "**3"
                                                elif str_name == "reciprocal":
                                                    str_ = "1/"
                                                elif str_name == "log":
                                                    str_ = "ln"
                                                else:
                                                    str_ = str_name
                                                # print(f"name {str_name}, {str_} in {ground_truth}")
                                                # print(str_ in ground_truth)
                                                # print()
                                                if str_ in ground_truth:
                                                    actual_primitives.append(str_name)

                                        # config["lib"]=[NAME_TO_KEY[key] for key in actual_primitives]
                                        config["lib"] = actual_primitives
                                        # append stacked functions
                                        config["lib"] += list(
                                            permutations(config["lib"], 2)
                                        )
                                        config["lib"].append("x")
                                        # zero is not being explicitly fitted, but as a special case of x (0*x+b)
                                        # config["lib"].append("0")

                                        # load dataset
                                        dataset_path = (
                                            dataset_folder / config["dataset"]
                                        )
                                        with open(dataset_path, "rb") as f:
                                            dataset = pickle.load(f)

                                        test_idx = np.random.choice(
                                            np.arange(len(dataset["train_input"])),
                                            size=config["val_size"],
                                            replace=False,
                                        )
                                        reversed_mask = np.ones(
                                            len(dataset["train_input"]), dtype=bool
                                        )
                                        reversed_mask[test_idx] = False

                                        # actually its val set
                                        dataset_ = {
                                            "train_input": torch.from_numpy(
                                                dataset["train_input"][reversed_mask]
                                            ),
                                            "train_label": torch.from_numpy(
                                                dataset["train_label"][reversed_mask]
                                            ),
                                            "test_input": torch.from_numpy(
                                                dataset["train_input"][test_idx]
                                            ),
                                            "test_label": torch.from_numpy(
                                                dataset["train_label"][test_idx]
                                            ),
                                        }

                                        # print(dataset["train_input"].shape)

                                        print(config["lib"])

                                        print(
                                            f"--------------------{path}--------------------"
                                        )
                                        print(
                                            f"Grid Condition {count} out of {gridsize}"
                                        )

                                        print(ground_truth)
                                        t0 = time.perf_counter()

                                        # try:
                                        (
                                            monitor,
                                            model,
                                            predicted_equation,
                                            outcome,
                                        ) = train_kan(config, dataset_, tolerance=0.9)

                                        '''except RuntimeError as e:
                                            print(e)
                                            monitor = SimpleNamespace()

                                            monitor.train_loss = np.array([np.nan])
                                            monitor.test_loss = np.array([np.nan])
                                            monitor.reg = np.array([np.nan])
                                            predicted_equation = sp.S.Zero
                                            outcome = "failed"'''
                                        runtime = time.perf_counter() - t0
                                        print(f"runtime: {runtime}")
                                        print("gt:", ground_truth)
                                        # print(type(predicted_equation))
                                        print("pred:", predicted_equation)

                                        symbols = list(
                                            sp.symbols([f"x_{i+1}" for i in range(n_v)])
                                        )
                                        # print(dataset["extrapolation_input"])
                                        # print(dataset["extrapolation_input"].shape)
                                        testloss_ext = success_metric(
                                            dataset["extrapolation_label"][:, 0],
                                            make_vectorized(
                                                predicted_equation, symbols
                                            )(dataset["extrapolation_input"]),
                                            norm=metric_norm,
                                        )
                                        testloss_int = success_metric(
                                            dataset["extrapolation_label"][:, 0],
                                            make_vectorized(
                                                predicted_equation, symbols
                                            )(dataset["extrapolation_input"]),
                                            norm=metric_norm,
                                        )

                                        success_val = testloss_ext
                                        print("success_val:", testloss_ext)

                                        run_logs.append(
                                            {
                                                "size": size,
                                                "train_loss": monitor.train_loss,
                                                "val_loss": monitor.test_loss,
                                                "test_loss": testloss_int,
                                                "reg": monitor.reg,
                                                "success_val": success_val,
                                                "prediction": str(predicted_equation),
                                                "runtime": runtime,
                                                "outcome": outcome,
                                            }
                                        )

                                        """for (key, val) in run_logs[-1].items():
                                            print(key, type(val))"""

                                        """print(success_val)
                                        print(type(success_val))
                                        print(success_val < success_threshold)
                                        print(monitor.test_loss)
                                        print(run_logs[-1]["test_loss"])"""

                                        if success_val < success_threshold:
                                            all_failed = False
                                            run_logs[-1]["success"] = True
                                            print("first success!!!")
                                            break

                                        # print("never reach this")
                                        run_logs[-1]["success"] = False

                                    trial_log = {
                                        "hyperparameters": {
                                            "grid": config["grid"],
                                            "k": config["k"],
                                            "structure_steps": config[
                                                "structure_steps"
                                            ],
                                            "lamb_l1": config["lamb_l1"],
                                            "lamb_entropy": config["lamb_entropy"],
                                            "loss_fn": config["loss_fn"],
                                            "lr": config["lr"],
                                            "symbolification_steps": config[
                                                "symbolification_steps"
                                            ],
                                            "lib": config["lib"],
                                            "finetune_steps": config["finetune_steps"],
                                            "n_1d_fits": config["n_1d_fits"],
                                        },
                                        "mult": multiplication,
                                        "prior": prior_knowledge,
                                        "lin": linear_transformations,
                                        "k": k,
                                        "nv": n_v,
                                        "comb": combination,
                                        "seed": seed,
                                        "run_logs": run_logs,
                                        "train_loss": run_logs[-1]["train_loss"][-1],
                                        "inter_loss": run_logs[-1]["test_loss"],
                                        "extra_loss": run_logs[-1]["success_val"],
                                        "success": run_logs[-1]["success"],
                                        "success_val": testloss_ext,
                                        "prediction": str(predicted_equation),
                                        "gt": ground_truth,
                                    }
                                    # print(trial_log["success"])
                                    grid_dict = config_kan.copy()
                                    grid_dict.pop("primitives")
                                    rootdir = Path(f"{folder}")
                                    dirname = Path(
                                        f"{'-'.join([str(key)+'_'+str(value) for key, value in grid_dict.items()])}"
                                    )
                                    fullpath = (
                                        rootdir
                                        / dirname
                                        / f"mult_{multiplication}-prior_{prior_knowledge}-lin_{linear_transformations}/k_{k}-nv_{n_v}-comb_{combination}-seed_{seed}.json"
                                    )
                                    fullpath.parent.mkdir(parents=True, exist_ok=True)
                                    with fullpath.open("wt") as f:
                                        json.dump(
                                            trial_log, f, cls=NumericTensorJSONEncoder
                                        )

                        if all_failed:
                            break

    # print(count)
    # print(path)


if __name__ == "__main__":
    main()

import json
import logging
import random
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


from src.train_pysr import train_pysr
from src.utils import configure_logging

# print(train)
# import src.utils

log_level = logging.DEBUG
# log_level = logging.INFO
overwrite_log = True
logger = logging.getLogger(__name__)


continue_params = {
    "multiplication": False,
    "prior_knowledge": False,
    "linear_transformations": False,
    "k": 3,
    "n_v": 3,
    "combination": 1,
    "seed": 0,
    # "count": 1,
}

continue_params = {}


continue_after = False


# dataset_folder = "datasets"
folder = "Results_pysr"
success_metric = nrmse
metric_norm = "std"
success_threshold = 0.2
# for debugging fast-forward training
scale_steps = 1.0  # / 100  # /100
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


def create_dict(**kwargs):
    return kwargs


config = create_dict(
    maxsize=20,  # max size of any equation/graph
    population_size=27,  # default 27
    populations=32,  # default=32
    niterations=5,  # 100,  # < Increase me for better results
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "log",
        "exp",
        "square",
        "cube",
        "inv(x) = 1/x",
        "sin",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    model_selection="loss",
    should_simplify=False,
    deterministic=True,  # gotta checkout runtime/parallelization impact
    random_state=42,
    parallelism="serial",
    alpha=5.0,  # (soe sort of) temperature, default = 3.17,
    annealing=True,
)

# sizes=[1,2,3,4,5]

# conditions
c_multiplication = (False, True)
c_prior_knowledge = (False,)  # (True, False)
c_linear_transformations = (False, True)

# for simulating results for debugging
test_rng = np.random.default_rng(42)


def main():
    global continue_params
    try:
        count = 0

        count += 1

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

                                try:
                                    with open(
                                        dataset_folder / f"equations_k{k}_nv{n_v}.txt",
                                        "rt",
                                    ) as f:
                                        equations = [line.rstrip() for line in f]
                                except FileNotFoundError:
                                    # some samples were rejected...
                                    # wait, only some combinations
                                    continue
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
                                    --------------------Results_pysr/mult_False-prior_False-lin_False/k_3-nv_3-comb_1-seed_0.json--------------------                                    Grid Condition 1 out of 256
                                    """

                                    local_vars = locals()

                                    # here
                                    if all(
                                        list(
                                            local_vars[key] == val
                                            for key, val in continue_params.items()
                                        )
                                    ):
                                        if continue_after:
                                            continue_params = dict()
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

                                    # safety seed setting
                                    np.random.seed(seed)
                                    torch.random.manual_seed(seed)
                                    random.seed(seed)

                                    # config

                                    actual_primitives = config["unary_operators"].copy()

                                    # not supposed to happen in this setup
                                    if prior_knowledge:
                                        safdsfds
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

                                    # zero is not being explicitly fitted, but as a special case of x (0*x+b)
                                    # config["lib"].append("0")

                                    # load dataset
                                    dataset_path = (
                                        dataset_folder
                                        / f"datasets_k{k}_nv{n_v}/{combination}.pkl"
                                    )
                                    try:
                                        with open(dataset_path, "rb") as f:
                                            dataset = pickle.load(f)
                                    except FileNotFoundError:
                                        # some samples were rejected...
                                        continue
                                    # print(dataset["train_input"].shape)

                                    print(
                                        f"--------------------{path}--------------------"
                                    )

                                    print(ground_truth)
                                    t0 = time.perf_counter()

                                    # here training and everything
                                    top5 = train_pysr(config, dataset, seed=seed)
                                    runtime = time.perf_counter() - t0

                                    for place, (
                                        predicted_equation,
                                        train_loss,
                                        score,
                                    ) in enumerate(top5):
                                        # try:
                                        '''except RuntimeError as e:
                                        print(e)
                                        monitor = SimpleNamespace()

                                        monitor.train_loss = np.array([np.nan])
                                        monitor.test_loss = np.array([np.nan])
                                        monitor.reg = np.array([np.nan])
                                        predicted_equation = sp.S.Zero
                                        outcome = "failed"'''
                                        print(f"place: {place}")
                                        print(f"runtime: {runtime}")
                                        print("gt:", ground_truth)
                                        # print(type(predicted_equation))
                                        print("pred:", predicted_equation)

                                        # here pysr naming convention
                                        symbols = list(
                                            sp.symbols([f"x{i}" for i in range(n_v)])
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
                                                "place": place,
                                                "train_loss": train_loss,
                                                "test_loss": testloss_int,
                                                "success_val": success_val,
                                                "prediction": str(predicted_equation),
                                                "runtime": runtime,
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

                                        # print("never reach this")
                                        run_logs[-1]["success"] = False

                                    trial_log = {
                                        "hyperparameters": config,
                                        "mult": multiplication,
                                        "prior": prior_knowledge,
                                        "lin": linear_transformations,
                                        "k": k,
                                        "nv": n_v,
                                        "comb": combination,
                                        "seed": seed,
                                        "run_logs": run_logs,
                                        "train_loss": run_logs[0]["train_loss"],
                                        "inter_loss": run_logs[0]["test_loss"],
                                        "extra_loss": run_logs[0]["success_val"],
                                        "success": run_logs[0]["success"],
                                        "success_val": run_logs[0]["success_val"],
                                        "prediction": str(predicted_equation),
                                        "gt": ground_truth,
                                    }
                                    # print(trial_log["success"])
                                    rootdir = Path(f"{folder}")

                                    fullpath = (
                                        rootdir
                                        / f"mult_{multiplication}-prior_{prior_knowledge}-lin_{linear_transformations}/k_{k}-nv_{n_v}-comb_{combination}-seed_{seed}.json"
                                    )
                                    fullpath.parent.mkdir(parents=True, exist_ok=True)
                                    with fullpath.open("wt") as f:
                                        json.dump(
                                            trial_log,
                                            f,
                                            cls=NumericTensorJSONEncoder,
                                        )

                        if all_failed:
                            break
    except Exception:

        logger.exception("Unhandled exception occurred")
        raise
    # print(count)
    # print(path)


if __name__ == "__main__":
    main()

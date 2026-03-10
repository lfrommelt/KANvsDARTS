from itertools import chain
import logging
from types import SimpleNamespace
import matplotlib.pyplot as plt

# from kan.MultKAN import KAN
from src.kan_sr import KAN_SR
import numpy as np

import torch
import math
import warnings


logger = logging.getLogger(__name__)


def safe_fit(kan_model, dataset, **kwargs):
    outcome = "normal"
    checkpoint = f"{kan_model.round}.{kan_model.state_id}"
    log = kan_model.fit(dataset, singularity_avoiding=False, **kwargs)
    if np.isnan(log["train_loss"][-1]):
        outcome = "safe"
        print("nan detected, switchig to safe mode")
        kan_model = kan_model.rewind(checkpoint)
        print(f"{kan_model.round}.{kan_model.state_id}")
        log = kan_model.fit(dataset, singularity_avoiding=True, **kwargs)
    return log, outcome

def find_candidates(model, dataset, config, logs, max_trials=6):

    # assumes:
    # - model (pretrained, unpruned) already exists
    # - dataset, logs, max_trials (e.g. max_trials = 6) already defined
    # - model has: .round, .state_id, .rewind(checkpoint), .prune_edge(threshold=...)
    # - model.fit() appends logs with ["test_loss"][-1] as validation loss
    # - model.edge_scores after model(dataset["train_input"]); model.attribute
    pretraining_logs=logs

    results = {
        "indices": [],     # grid indices in evaluation order
        "thresholds": [],  # actual thresholds T_j
        "log_thresholds": [],  # corresponding g_j
        "val_losses": [],
        "models": [],
        "logs": [],
    }

    # 1) Remember checkpoint of the unpruned model
    checkpoint = f"{model.round}.{model.state_id}"

    # 2) Get edge scores and build log-score vector
    _ = model(dataset["train_input"])
    model.attribute()

    edge_scores_per_layer = model.edge_scores
    all_scores = torch.cat([s.flatten() for s in edge_scores_per_layer])

    eps = 1e-12
    all_scores = all_scores.clamp_min(eps)
    log_s = torch.log(all_scores)

    # sort descending
    log_s_sorted, _ = torch.sort(log_s, descending=True)


    # remove duplicates to avoid degenerate grid points
    log_s_unique = torch.unique(log_s_sorted)
    # sort unique values in descending order explicitly
    log_s_unique, _ = torch.sort(log_s_unique, descending=True)

    if log_s_unique.numel() < 2:
        warnings.warn("Not enough distinct scores to build a grid. Aborting pruning search.")
        # nothing more to do
    else:
        # 3) Build grid: midpoints between neighboring log scores
        #    log_s_unique: length M  =>  grid g_j: length G = M - 1
        g = 0.5 * (log_s_unique[:-1] + log_s_unique[1:])
        G = g.numel()  # number of grid points

        # 4) Initial 3 indices via top-3 log-gaps
        log_gaps = log_s_unique[:-1] - log_s_unique[1:]   # same index range as g
        k = min(3, log_gaps.numel())
        if k == 0:
            warnings.warn("No gaps available to define thresholds. Aborting pruning search.")
        else:
            top_gap_vals, top_gap_idx = torch.topk(log_gaps, k, largest=True, sorted=True)
            init_indices = sorted(top_gap_idx.cpu().tolist())

            evaluated_indices = []  # in evaluation order
            evaluated_set = set()   # for fast membership

            # helper: evaluate model at grid index j
            def evaluate_index(j):
                """Rewind, prune at T_j, train, log loss, store result; return loss, model."""
                logs=pretraining_logs.copy()
                T = math.exp(g[j].item())
                m = model.rewind(checkpoint)
                _ = m(dataset["train_input"])
                m.attribute()

                # keeps layers alive, so some grid values might be functional duplicates
                m = m.prune_edge(threshold=T)
                logger.info(f"KAN after pruning at idx {j} with thresh {T}:")
                m.log_self()
                logs.append(m.fit(
                    dataset,
                    opt="LBFGS",
                    steps=config["symbolification_steps"],
                    log=1,
                    lamb=0.0,
                    lamb_l1=0.0,
                    lamb_entropy=0.0,
                    lamb_coef=0.0,
                    lamb_coefdiff=0.0,
                    update_grid=True,
                    grid_update_num=10,
                    loss_fn=config["loss_fn"],
                    lr=config["lr"],
                    start_grid_update_step=-1,
                    stop_grid_update_step=50,
                    batch=-1,
                    y_th=1000.0,
                    reg_metric="edge_forward_spline_n",
                    display_metrics=None,
                    verbose=False,
                    print_gradients=False,
                ))
                L = logs[-1]["test_loss"][-1]
                results["indices"].append(j)
                results["thresholds"].append(T)
                results["log_thresholds"].append(g[j].item())
                results["val_losses"].append(float(L))
                results["models"].append(m)
                results["logs"].append(logs)
                evaluated_indices.append(j)
                evaluated_set.add(j)
                return L, m

            # 5) Evaluate initial indices
            for j in init_indices:
                logger.debug(f"evaluating threshold candidate {j}/{init_indices}")
                if len(evaluated_indices) >= max_trials:
                    break
                if j in evaluated_set:
                    continue
                evaluate_index(j)

            # If we got less than 3 distinct points, we can't fit a quadratic robustly
            if len(evaluated_indices) < 3:
                warnings.warn("Fewer than 3 evaluated thresholds; skipping quadratic refinement.")
            else:
                # region_indices: 3 points used for fitting
                region_indices = sorted(evaluated_indices[:3])  # seed with the first 3 evaluated

                # main refinement loop
                while len(evaluated_indices) < max_trials:
                    # ensure we have at least 3 for region
                    if len(region_indices) < 3:
                        # if not enough, just exit
                        break

                    # gather (x, y) for current region
                    xs = torch.tensor([g[i].item() for i in region_indices], dtype=torch.float64)
                    ys = torch.tensor(
                        [results["val_losses"][results["indices"].index(i)] for i in region_indices],
                        dtype=torch.float64,
                    )

                    # solve quadratic y = a x^2 + b x + c using 3 points analytically
                    # build Vandermonde and solve
                    X = torch.stack([xs**2, xs, torch.ones_like(xs)], dim=1)  # [3,3]
                    try:
                        theta = torch.linalg.solve(X, ys)  # [3], a, b, c
                        a, b, c = theta.tolist()
                    except RuntimeError:
                        warnings.warn("Quadratic fit failed (singular matrix). Stopping refinement.")
                        break

                    # determine new candidate index
                    if a > 0:
                        # convex: analytic minimum
                        x_star = -b / (2.0 * a)

                        # handle out-of-bounds: take closest unused grid point at ends
                        if x_star < g[0].item():
                            # search from left end for unused index
                            candidate = None
                            for j in range(G):
                                if j not in evaluated_set:
                                    candidate = j
                                    break
                            if candidate is None:
                                # all grid points already evaluated
                                break
                            i_new = candidate
                            x_target = g[i_new].item()  # use its location as region target
                        elif x_star > g[-1].item():
                            # search from right end for unused index
                            candidate = None
                            for j in range(G - 1, -1, -1):
                                if j not in evaluated_set:
                                    candidate = j
                                    break
                            if candidate is None:
                                break
                            i_new = candidate
                            x_target = g[i_new].item()
                        else:
                            # inside global grid: find nearest unused grid index to x_star
                            # start from closest j and expand outward until find unused
                            # compute distances to all grid points
                            dists = torch.abs(g - x_star)
                            sorted_idx = torch.argsort(dists)
                            i_new = None
                            for j in sorted_idx.tolist():
                                if j not in evaluated_set:
                                    i_new = j
                                    break
                            if i_new is None:
                                # no unused grid points left
                                break
                            x_target = x_star
                    else:
                        # a <= 0: fallback; direction based on sign of b
                        warnings.warn("Quadratic fit not convex (a <= 0); using directional fallback.")
                        # find best index in region (smallest loss)
                        region_losses = [
                            results["val_losses"][results["indices"].index(i)] for i in region_indices
                        ]
                        best_local_idx = region_indices[
                            int(torch.argmin(torch.tensor(region_losses)).item())
                        ]

                        # choose direction
                        if b > 0:
                            # move left from best_local_idx
                            search_order = list(range(best_local_idx - 1, -1, -1))
                        elif b < 0:
                            # move right
                            search_order = list(range(best_local_idx + 1, G))
                        else:
                            # b == 0: ambiguous, search both sides by distance
                            left = list(range(best_local_idx - 1, -1, -1))
                            right = list(range(best_local_idx + 1, G))
                            # interleave left and right
                            search_order = []
                            l_ptr, r_ptr = 0, 0
                            while l_ptr < len(left) or r_ptr < len(right):
                                if l_ptr < len(left):
                                    search_order.append(left[l_ptr])
                                    l_ptr += 1
                                if r_ptr < len(right):
                                    search_order.append(right[r_ptr])
                                    r_ptr += 1

                        i_new = None
                        for j in search_order:
                            if j not in evaluated_set:
                                i_new = j
                                break
                        if i_new is None:
                            # no unused index in this direction(s)
                            break
                        # for region selection, target is the grid location we actually moved to
                        x_target = g[i_new].item()

                    # if new index already evaluated (extremely unlikely given checks), stop
                    if i_new in evaluated_set:
                        break

                    # 6) Evaluate new index
                    if len(evaluated_indices) >= max_trials:
                        break
                    
                    logger.debug(f"evaluating threshold candidate {i_new}/{region_indices}")
                    evaluate_index(i_new, )

                    # 7) Update region_indices: 3 evaluated points closest to x_target
                    idxs_sorted = sorted(evaluated_indices)
                    # distances to target
                    dists_region = [(abs(g[j].item() - x_target), j) for j in idxs_sorted]
                    dists_region.sort(key=lambda x: x[0])
                    region_indices = [j for _, j in dists_region[:3]]

    # results dict now contains:
    # - "indices": evaluated grid indices (order of evaluation)
    # - "thresholds": corresponding T_j
    # - "log_thresholds": g_j values
    # - "val_losses": validation losses
    # - "models": pruned+finetuned models
    return results


def train_kan_vanilla(config, dataset, tolerance=0.9):
    outcome = "default"
    logs = []
    model = KAN_SR(
        width=config["width"],
        grid=config["grid"],
        k=config["k"],
        mult_arity=0,
        noise_scale=0.3,
        scale_base_mu=0.0,
        scale_base_sigma=1.0,
        base_fun="zero",
        symbolic_enabled=True,
        affine_trainable=False,
        grid_eps=0.02,
        grid_range=[0, 5.0],
        sp_trainable=False,
        sb_trainable=False,
        seed=config["seed"],
        save_act=True,
        sparse_init=False,
        auto_save=True,
        first_init=True,
        ckpt_path="./model",
        # state_id=0,
        # round=0,
        device="cpu",
        real_affine_trainable=0.0,
    )
    # todo: tolerance gotta be normal hp in config
    eps = 1e-10
    ## Structure Phase
    singularity_avoiding = False

    checkpoint = f"{model.round}.{model.state_id}"
    print("current checkpoint:", checkpoint)
    logs.append(
        model.fit(
            dataset,
            opt="LBFGS",
            steps=config["structure_steps"],
            log=1,
            lamb=0.0,
            lamb_l1=config["lamb_l1"],
            lamb_entropy=config["lamb_entropy"],
            lamb_coef=0.0,
            lamb_coefdiff=0.0,
            update_grid=True,
            grid_update_num=10,
            loss_fn=config["loss_fn"],
            lr=config["lr"],
            start_grid_update_step=-1,
            stop_grid_update_step=50,
            batch=-1,
            metrics=None,
            save_fig=False,
            in_vars=None,
            out_vars=None,
            beta=3,
            save_fig_freq=1,
            singularity_avoiding=singularity_avoiding,
            y_th=1000.0,
            reg_metric="edge_forward_spline_n",
            display_metrics=None,
            verbose=False,
            print_gradients=False,
        )
    )

    # forward pass for pruning values
    model(dataset["train_input"])
    model.attribute()

    logger.info(f"KAN before pruning:")
    model.log_self()

    pruning_results = find_candidates(model, dataset=dataset, logs=logs, config=config)


    # model.plot()
    # plt.show()

    steps = 0
    ## Symbolification Phase
    #while sum([act_fun.mask.sum() for act_fun in model.act_fun]):+

    best_loss=np.inf
    # will default to arbitrary nan model if all models are nan
    best_idx=-1

    for candidate_idx in range(len(pruning_results["models"])):
        model=pruning_results["models"][candidate_idx]
        logger.info(f"---symbolification step {candidate_idx}/{len(pruning_results["models"])}---")
        log = model.fit(
            dataset,
            opt="LBFGS",
            steps=config["symbolification_steps"],
            log=1,
            lamb=0.0,
            lamb_l1=0.0,
            lamb_entropy=0.0,
            lamb_coef=0.0,
            lamb_coefdiff=0.0,
            update_grid=True,
            grid_update_num=10,
            loss_fn=config["loss_fn"],
            lr=config["lr"],
            start_grid_update_step=-1,
            stop_grid_update_step=50,
            batch=-1,
            y_th=1000.0,
            reg_metric="edge_forward_spline_n",
            display_metrics=None,
            verbose=False,
            print_gradients=False,
        )
        logger.info(f"Outcome: {outcome}")

        pruning_results["logs"][candidate_idx].append(log)
        model(dataset["train_input"])
        # we exploit that setting thresh to inf makes us fit all edges in one go
        model.minimal_auto_symbolic(
            verbose=False, lib=config["lib"], n_1d_fits=config["n_1d_fits"], r2_threshold=torch.inf,
        )
        # print(*[act_fun.mask for act_fun in model.act_fun], sep="\n")

        ## Finetune
        logger.debug(f"prediction before finetune:\n{model.symbolic_formula(simplify=False)[0][0]}")
        log, outcome = safe_fit(
            model,
            dataset,
            opt="LBFGS",
            steps=config["finetune_steps"],
            log=1,
            lamb=0.0,
            lamb_l1=0.0,
            lamb_entropy=0.0,
            lamb_coef=0.0,
            lamb_coefdiff=0.0,
            update_grid=True,
            grid_update_num=10,
            loss_fn=config["loss_fn"],
            lr=config["lr"],
            start_grid_update_step=-1,
            stop_grid_update_step=50,
            batch=-1,
            y_th=1000.0,
            reg_metric="edge_forward_spline_n",
            display_metrics=None,
            verbose=False,
            print_gradients=False,
        )
        pruning_results["logs"][candidate_idx].append(log)
        current_loss=log["test_loss"][-1]
        if current_loss < best_loss:
            best_loss = current_loss
            best_idx=candidate_idx
        
    
    # for now lets forget all but the best one
    model=pruning_results["models"][best_idx]
    logs=pruning_results["logs"][best_idx]

    """logs.append(model.fit(dataset, opt="LBFGS", steps=config["finetune_steps"], 
                            log=1, lamb=0., lamb_l1=0.0, lamb_entropy=0.0, 
                            lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, 
                            loss_fn=config["loss_fn"], lr=config["lr"], start_grid_update_step=-1,
                            stop_grid_update_step=50, batch=-1, metrics=None, save_fig=False, 
                            in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video',
                            singularity_avoiding=singularity_avoiding, y_th=1000., 
                            reg_metric='edge_forward_spline_n', display_metrics=None, 
                            verbose=False, print_gradients=False,
                            )
                )"""

    # print("mask",*[layer.mask for layer in model.act_fun], sep="\n")
    # print("symbolic_mask",*[layer.mask for layer in model.symbolic_fun], sep="\n")

    # sympy tends to get stuck in some sort of endless recursion sometimes
    predicted_equation = model.symbolic_formula(simplify=False)[0][0]  # _light()

    #print("----keys here----")
    #print(logs[0].keys())
    #print(logs[0])
    #print(logs[1])
    
    log = {
        key: list(chain.from_iterable(d[key] for d in logs))
        for key in logs[0]  # assume at least one dict
    }

    logger.info(f"\nKAN prediction:\n\n{predicted_equation}\n")
    logger.info(f"With architecture:")
    model.log_self(symbolic=True, sparsity=False)
    monitor = SimpleNamespace(**log)


    logger.info("\n")
    model.log_self(symbolic=True, sparsity=True)
    return monitor, model, predicted_equation, outcome

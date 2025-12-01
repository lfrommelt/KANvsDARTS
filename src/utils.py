import json
from tqdm import tqdm
import itertools

# from equation_tree import instantiate_constants
import numpy as np
import torch
import os
import sys
import sklearn
import copy
from sortedcontainers import SortedSet
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import random
import types
import matplotlib as mpl


import sympy as sp


class NanError(Exception):
    pass


@dataclass
class Monitor:
    """
    No callback, just use for storing traing curves
    """

    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    test_loss: list = field(default_factory=list)
    alphas: list = field(default_factory=list)
    coefficients: torch.Tensor = torch.zeros(1)


class ContextClass:
    """
    Simple class just for enabling static attributes as a safe and
    unambigous way of having global variables
    """

    gumbel_softmax_enabled = False
    discrete_enabled = False


def sample_and_replace(text: str, seed: int) -> str:
    """
    Replaces exactly one plus sign ('+') in `text` with an asterisk ('*').

    The index of the plus sign to be replaced is drawn from a categorical
    (discrete) uniform distribution over 0 … (n_plus-1), where n_plus is the
    number of plus signs in the input.  The PRNG is initialised with `seed`
    so results are reproducible.

    Parameters
    ----------
    seed : int
        Seed for NumPy’s random number generator.
    text : str
        Any string that contains one or more '+' characters.

    Returns
    -------
    str
        A copy of `text` in which exactly one '+' has been changed to '*'.
        If `text` contains no '+', the original string is returned unchanged.
    """

    # Locate every '+' in the string
    plus_positions = [idx for idx, char in enumerate(text) if char == "+"]

    # Nothing to replace
    if not plus_positions:
        return text

    n_plus = len(plus_positions)

    # Uniform categorical distribution over {0, …, n_plus-1}
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.arange(n_plus))

    # Replace the chosen '+' with '*'
    text_list = list(text)
    text_list[plus_positions[chosen]] = "*"

    return "".join(text_list)


def wrap_expression(expr, seed):
    values = np.linspace(0.01, 2.00, 200)
    rng = np.random.Generator(np.random.PCG64(seed))

    def insert_f_coefficient(match):
        return f"{rng.choice(values):.2f}*f"

    def insert_bias_after_paren(match):
        return f"{match.group(0)} + {rng.choice(values):.2f}"

    def wrap_x(match):
        var = match.group(0)  # e.g., "x_2"
        c = rng.choice(values)
        b = rng.choice(values)
        return f"({c:.2f}*{var} + {b:.2f})"

    # Step 1: Insert coefficient before 'f'
    expr = re.sub(r"f", insert_f_coefficient, expr)

    # Step 2: Insert bias after ')'
    expr = re.sub(r"$", insert_bias_after_paren, expr)

    # Step 3: Wrap affine transform around indexed variables
    expr = re.sub(r"x_\d+", wrap_x, expr)

    return expr


def testplot(model, datasets, name, metric="nrmse", metric_norm="range", x_axis=0):

    if metric == "rmse":
        eval_ = rmse
    elif metric == "r2":
        eval_ = r2
    elif metric == "nrmse":
        eval_ = nrmse
    elif metric == "rmspe":
        eval_ = rmspe
    elif metric == "sl_rmse":
        eval_ = sl_rmse
    fig, axes = plt.subplots(1, 1, figsize=(15, 4), sharey=True)
    ax = axes
    # plot prediction curve
    # y_true = gt(x_plot)

    # scatter data
    for label, xx, yy in datasets:
        # print(xx[:10])
        # print(label)
        if label == "train":
            m, c = "o", "C0"
            # shade training domain
            ax.axvspan(xx.min(), xx.max(), color="gray", alpha=0.2)
        elif "inter" in label:
            m, c = "s", "C1"
        elif "extra" in label:
            m, c = "*", "C2"
            """elif "pred" in label:
            m, c = '+', 'C3"""
        else:
            print(label)
            sgdsag
        ax.scatter(xx[:, x_axis], yy, marker=m, c=c, label=label)

    allx = np.concatenate([datasets[i][1] for i in range(3)], axis=0)

    # print(*[datasets[i][2][:5] for i in range(3)], sep="\n")
    ally = np.concatenate([datasets[i][2] for i in range(3)], axis=0)

    # print(model(allx))
    # ax.plot(allx, ally, 'k-', lw=1, label="gt")
    ax.scatter(
        allx[:, x_axis], model(allx), lw=1, label="prediction", marker="+", c="C3"
    )

    # compute & display RMSEs
    txt = ""
    for lbl, xx, yy in datasets:
        # print(lbl)
        r = eval_(yy, model(xx), norm=metric_norm)
        # print("----")
        txt += f"{lbl[:5]} {metric}={r:.3f}\n"
    ax.text(
        0.05,
        0.95,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.1),
    )

    ax.set_title(name)
    ax.set_xlim(-10, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


def testplot2(
    model, datasets, name, metric="nrmse", metric_norm="range", x_axis=0, s=72, f2=32
):

    mpl.rcParams.update({"font.size": 32})

    if metric == "rmse":
        eval_ = rmse
    elif metric == "r2":
        eval_ = r2
    elif metric == "nrmse":
        eval_ = nrmse
    elif metric == "rmspe":
        eval_ = rmspe
    elif metric == "sl_rmse":
        eval_ = sl_rmse
    fig, axes = plt.subplots(1, 1, figsize=(30, 8), sharey=True)
    ax = axes
    # plot prediction curve
    # y_true = gt(x_plot)

    # scatter data
    for label, xx, yy in datasets:
        # print(xx[:10])
        # print(label)
        if label == "train":
            m, c = "o", "C0"
            # shade training domain
            ax.axvspan(xx.min(), xx.max(), color="gray", alpha=0.2)
        elif "inter" in label:
            m, c = "s", "C1"
        elif "extra" in label:
            m, c = "*", "C2"
            """elif "pred" in label:
            m, c = '+', 'C3"""
        else:
            print(label)
            sgdsag
        ax.scatter(xx[:, x_axis], yy, marker=m, c=c, label=label, s=s)

    allx = np.concatenate([datasets[i][1] for i in range(3)], axis=0)

    # print(*[datasets[i][2][:5] for i in range(3)], sep="\n")
    ally = np.concatenate([datasets[i][2] for i in range(3)], axis=0)

    # print(model(allx))
    # ax.plot(allx, ally, 'k-', lw=1, label="gt")
    ax.scatter(
        allx[:, x_axis],
        model(allx),
        lw=1,
        label="prediction",
        marker="+",
        c="C3",
        s=s * 1.2,
    )

    # compute & display RMSEs
    txt = ""
    for lbl, xx, yy in datasets:
        # print(lbl)
        r = eval_(yy, model(xx), norm=metric_norm)
        # print("----")
        txt += f"{lbl[:5]} {metric}={r:.3f}\n"
    ax.text(
        0.05,
        0.95,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=f2,
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.1),
    )

    # ax.set_title(name)
    # ax.set_xlim(-10,10)
    ax.set_xlabel(f"x_{x_axis+1}")
    ax.set_ylabel("f(x)")
    ax.legend(loc="lower right")  # , fontsize=8)
    plt.tight_layout()
    plt.show()


class SamplingError(ValueError):
    """Raised when a sampling procedure fails or produces invalid data."""

    pass


def round_and_simplify(expr, ndigits=1):
    """
    Round all Float numbers occurring in `expr` to *ndigits* decimals and
    return a simplified expression (0*anything → 0, etc.).

    Parameters
    ----------
    expr : str | sympy.Expr
        The expression to process.
    ndigits : int, optional
        Number of decimal places to keep (default: 1).

    Returns
    -------
    sympy.Expr
    """
    # expr = sympify(expr)                # ensure we have a SymPy expression

    def _round_float(f):
        """Round one sympy.Float instance to ndigits decimals."""
        v = round(float(f), ndigits)  # Python float → rounded
        if abs(v) < 10 ** (-ndigits):  # get rid of -0.0 / tiny remainders
            v = 0
        return sp.Float(v)

    # replace every Float in the expression by its rounded counterpart
    rounded = expr.xreplace({f: _round_float(f) for f in expr.atoms(sp.Float)})

    # final clean-up
    return simplify(rounded)


x_1, x_2, x_3 = sp.symbols("x_1 x_2 x_3")
_ALLOWED = [x_1, x_2, x_3]

'''def make_vectorized(expr, inputs=None):
    """
    expr    : a SymPy expression or a bare Python int/float
    inputs  : list of symbols (subset of [x_1,x_2,x_3]) that
              you want to use as the columns of X.
              If None, defaults to sorted(expr.free_symbols).
    Returns : f(X) which expects X.shape = (n_points, len(inputs))
              and returns a 1-D numpy array of length n_points.
    """

    # 0) force expr to be a SymPy object so .free_symbols always exists
    expr = sp.sympify(expr)

    # 1) pick or validate your input‐symbols
    if inputs is None:
        # pick out expr.free_symbols in the order x_1,x_2,x_3
        inputs = [s for s in _ALLOWED if s in expr.free_symbols]
    else:
        inputs = list(inputs)
        bad = set(inputs) - set(_ALLOWED)
        if bad:
            raise ValueError(f"Unsupported symbols in inputs: {bad}")
    n_in = len(inputs)
    if n_in > 3:
        raise ValueError("Can't have more than 3 input symbols")

    # 2) ensure expr mentions only those inputs
    extra = expr.free_symbols - set(inputs)
    if extra:
        raise ValueError(f"Expression uses symbols {extra} not in inputs")

    # 3) lambdify to a numpy‐aware function
    f_np = sp.lambdify(inputs, expr, modules='numpy')

    # 4) wrap so that we feed it columns of X
    def f(X):
        X = np.asarray(X)
        # allow passing a single-point as 1-D
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError("X must be 1-D or 2-D")
        n_pts, n_cols = X.shape
        if n_cols != n_in:
            raise ValueError(f"Expected {n_in} columns, got shape {X.shape}")

        # unpack columns
        args = [X[:, i] for i in range(n_in)]
        #print(args)
        out = f_np(*args)

        #print(out)
        # if expr was constant (n_in==0), out is scalar
        if np.isscalar(out) or (isinstance(out, np.ndarray) and out.ndim == 0):
            return np.full(n_pts, out, dtype=float)
        arr = np.asarray(out)
        # check arr.ndim==1 && arr.shape[0]==n_pts ...
        return arr

    return f'''


def make_vectorized(expr, inputs=None):
    """
    expr    : a SymPy expression or a bare Python int/float
    inputs  : list of symbols (subset of [x_1,x_2,x_3]) that
              you want to use as the columns of X.
              If None, defaults to sorted(expr.free_symbols).
    Returns : f(X) which expects X.shape = (n_points, len(inputs))
              and returns a 1-D numpy array of length n_points.
    """

    # accept the Zero-class itself as a constant 0
    if expr is sp.core.numbers.Zero or expr is None:
        expr = 0

    # 0) force expr to be a SymPy object so .free_symbols always exists
    expr = sp.sympify(expr)

    # 1) pick or validate your input‐symbols
    if inputs is None:
        inputs = [s for s in _ALLOWED if s in expr.free_symbols]
    else:
        inputs = list(inputs)
        bad = set(inputs) - set(_ALLOWED)
        if bad:
            raise ValueError(f"Unsupported symbols in inputs: {bad}")
    n_in = len(inputs)
    if n_in > 3:
        raise ValueError("Can't have more than 3 input symbols")

    """# 2) ensure expr mentions only those inputs
    extra = expr.free_symbols - set(inputs)
    if extra:
        raise ValueError(f"Expression uses symbols {extra} not in inputs")"""

    # 3) lambdify to a numpy‐aware function
    f_np = sp.lambdify(inputs, expr, modules="numpy")

    # 4) wrap so that we feed it columns of X
    def f(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError("X must be 1-D or 2-D")
        n_pts, n_cols = X.shape
        if n_cols != n_in:
            raise ValueError(f"Expected {n_in} columns, got shape {X.shape}")

        args = [X[:, i] for i in range(n_in)]
        out = f_np(*args)

        if np.isscalar(out) or (isinstance(out, np.ndarray) and out.ndim == 0):
            return np.full(n_pts, out, dtype=float)
        # print(out)
        return np.asarray(out)

    return f


def rmse(y_true, y_pred, norm=None):
    y_true_ = y_true[~np.isnan(y_pred)]
    y_pred_ = y_pred[~np.isnan(y_pred)]
    return np.sqrt(np.mean((y_true_ - y_pred_) ** 2))


def nrmse(y_true, y_pred, norm="range"):
    """
    Normalised root-mean-square error (scale-invariant).

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values (may contain NaNs).
    norm : {'range', 'mean', 'std'}, default 'range'
        • 'range' → divide by (max - min) of y_true
        • 'mean'  → divide by mean(y_true)  (CV-RMSE)
        • 'std'   → divide by std(y_true)

    Returns
    -------
    float
        The normalised RMSE.  np.nan if it cannot be computed.
    """
    # mask out the entries whose prediction is NaN
    mask = (~np.isnan(y_pred)) & (~np.isnan(y_true))
    y_t = np.asarray(y_true)[mask]
    y_p = np.asarray(y_pred)[mask]

    print(y_t.shape)
    print(y_p.shape)

    if y_t.size == 0:
        return np.nan  # nothing to evaluate

    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))

    if norm == "range":
        denom = y_t.max() - y_t.min()
    elif norm == "mean":
        denom = np.abs(np.mean(y_t))
    elif norm == "std":
        denom = np.std(y_t, ddof=0)
    else:
        raise ValueError("norm must be 'range', 'mean', or 'std'")

    if denom == 0:
        return np.nan  # scale is zero → undefined

    return rmse / denom


def sl_rmse_old(y_true, y_pred, lam=1.0, eps=1e-12, norm=None):
    mask = (~np.isnan(y_pred)) & (~np.isnan(y_true))
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]

    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)

    # protect against zeros and keep magnitudes
    y_abs = np.abs(y_true) + eps
    yp_abs = np.abs(y_pred) + eps

    # best vertical shift in log-space
    delta = np.mean(np.log(yp_abs) - np.log(y_abs))
    yp_adj = yp_abs * np.exp(-delta)

    # shape / magnitude mismatch
    err = np.log(yp_adj) - np.log(y_abs)

    # sign mismatch (0 if equal sign, 1 if opposite)
    sign_err = (sign_true != sign_pred).astype(float)

    return np.sqrt(np.mean(err**2 + lam * sign_err))


def sl_rmse(y_true, y_pred, lam=1.0, eps=1e-12, **kwargs):

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    sign_err = (np.sign(y_true) != np.sign(y_pred)).astype(float)

    log_y = np.log(np.abs(y_true) + eps)
    log_yp = np.log(np.abs(y_pred) + eps)

    delta = np.mean(log_yp - log_y)  # same as before
    err = (log_yp - delta) - log_y  # <-- no exp() at all

    return np.sqrt(np.mean(err**2 + lam * sign_err))


def rmspe(y_true, y_pred, norm=None, eps=1e-12):

    mask = (~np.isnan(y_pred)) & (~np.isnan(y_true))
    y_t = np.asarray(y_true)[mask]
    y_p = np.asarray(y_pred)[mask]

    # y_true = np.asarray(y_true, dtype=float)
    # y_pred = np.asarray(y_pred, dtype=float)
    rel_sq = ((y_p - y_t) / np.maximum(np.abs(y_t), eps)) ** 2
    return np.sqrt(np.mean(rel_sq))


def r2(y_true, y_pred, norm=None):
    """
    Coefficient of determination (R²) with the same NaN-handling strategy
    used in the `rmse` function above.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Predicted target values (may contain NaNs).

    Returns
    -------
    float
        R² score.  Returns np.nan if it cannot be computed (e.g. all
        predictions are NaN or `y_true` has zero variance).
    """
    # Keep only the entries whose prediction is not NaN
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    # print(y_true)
    # print(y_pred)
    y_true_ = y_true[mask]
    y_pred_ = y_pred[mask]
    """print(y_true_)
    print(y_pred_)
    print(y_true_-y_pred)
    print(rmse(y_true_, y_pred_))"""

    # Nothing left after masking → undefined R²
    if y_true_.size == 0:
        return np.nan

    ss_res = np.sum((y_true_ - y_pred_) ** 2)
    ss_tot = np.sum((y_true_ - np.mean(y_true_)) ** 2)

    # If variance is zero, R² is undefined
    if ss_tot == 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


@dataclass
class Dump:
    latent_states = list()


"""sys.path.append(
    os.path.dirname(os.getcwd())
    + "/merge_requests/deterministic_darts/autora-theorist-darts/src"
)"""
# print(sys.path)
# from autora.theorist.darts import DARTSExecutionMonitor, DARTSRegressor
# from autora.theorist.darts.regressor import _generate_model

# not nice, but we need all sympy symbolic functions in our namespace for eval
# from sympy import *
import sympy as sp
import re
import sympy as sp

# import equation_tree
"""from equation_tree.prior import (
    priors_from_space,
    structure_prior_from_max_depth,
    structure_prior_from_depth,
)"""
import itertools

OPERATORS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    # "^": lambda a, b: a**b,
    # "**": lambda a, b: a**b,
    # "max": lambda a, b: np.max([a, b]),
    # "min": lambda a, b: np.min([a, b]),
}

FUNCTIONS = {
    "sin": lambda a: np.sin(a),
    # "cos": lambda a: np.cos(a),
    # "tan": lambda a: np.tan(a),
    # "exp": lambda a: np.exp(a),
    # "log": lambda a: np.log(a),
    # "sqrt": lambda a: np.sqrt(a),
    # "abs": lambda a: np.abs(a),
    # "acos": lambda a: np.arccos(a),
    # "arg": lambda a: np.pi * a / 180,
    # "asin": lambda a: np.arcsin(a),
    # "sinh": lambda a: np.sinh(a),
    # "cosh": lambda a: np.cosh(a),
    # "tanh": lambda a: np.tanh(a),
    # "cubed": lambda a: a**3,
    "squared": lambda a: a**2,
    # "cot": lambda a: 1 / np.tan(a),
}

PROBLEM_DOMAINS = {
    "depth": [3, 5],
    "max_vars": [2],
    "n_equations": [2],
    "re-samplings": [
        1
    ],  # must never have more than one item, re-sample with same prior
    "training_domain": [(-1.0, 1.0)],
}

PRIMITIVES = (
    "power_two",
    "power_three",
    "exp",
    "ln",
    "reciprocal",
    "sin",
)


def check_success(prediction, ground_truth, v=False, levels=3):
    # check success
    # darts var name convention
    # eq_=normalize_varnames(ground_truth)
    eq_ = str_to_sympy(ground_truth, var="x_")

    # catch cases, where predicted_equation is just a constant
    pred = ensure_symbolic(prediction)
    # rewrite cos(x) to sin(x+pi/2)
    pred = pred.rewrite(sp.sin)
    # evaluate with pi as float, so the previous step is not being reverted
    pred = pred.subs(sp.pi, sp.N(sp.pi))
    # sin(x+a) to sin(x+a%(2pi)), i.e. minimal positive additive argument
    pred = pred.replace(lambda f: f.func is sp.sin, norm_sin)
    # round, expand, replace equal values by symbols, simplify, drop values below sensitivity
    simpler_ = []
    if levels > 0:
        simpler_.append(simplify(pred, decimal_points=0, sensitivity=1e-4))
    if levels > 1:
        simpler_.append(simplify(pred, decimal_points=1, sensitivity=1e-4))
    if levels > 2:
        simpler_.append(simplify(pred, decimal_points=2, sensitivity=1e-4))

    eq_ = unify_float_precision(eq_)

    success = False
    for simpler in simpler_:
        # 1.0*x to x
        simpler = remove_unit_coeff(simpler)
        # make sure that floats have the same precision so they can be truly equal
        simpler = unify_float_precision(simpler)

        success = simpler.equals(eq_)
        if v:
            print(f"{simpler}=={eq_}")
            print(success)
            print()
        if success:
            break

    if success is None:
        success = False
    return success


def plot_alphas_fresh(monitor, n_var):
    alphas = monitor.alphas
    for node in range(len(alphas[0])):
        for input_ in range(len(alphas[0][node])):
            node_size = len(alphas[0][node])
            from_ = (
                f"x_{input_-node_size+n_var}"
                if input_ >= node_size - n_var
                else f"h_{input_}"
            )
            to = f"h_{node}"
            plt.title(f"{from_}->{to}")
            for i, primitive in enumerate(alphas[0][0][0].keys()):
                plt.plot(
                    np.arange(len(alphas)),
                    [alphas[j][node][input_][primitive] for j in range(len(alphas))],
                    label=str(primitive),
                )
            plt.legend()
            plt.show()


def plot_alphas_old(monitor, n_var, primitives):
    alphas = monitor.alphas

    i = 0
    break_ = false
    while not break_:
        i += 1
        overall_i = 0
        for input_ in range(i):
            node_size = i
            from_ = (
                f"x_{input_-node_size+n_var}"
                if input_ >= node_size - n_var
                else f"h_{input_}"
            )
            to = f"h_{i}"
            plt.title(f"{from_}->{to}")
            for k, primitive in enumerate(primitives):
                plt.plot(
                    np.arange(len(alphas)),
                    [alphas[l][overall_i][k] for l in range(len(alphas))],
                    label=primitive,
                )
            plt.legend()
            plt.show()
            overall_i += 1


def normalize_varnames(s):
    return re.sub(r"x_(\d+)", r"x\1", s)


def plot_pair_histograms(rectangles, n):
    # Build all possible pairs
    ordered_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    unordered_pairs = [tuple(sorted((i, j))) for i in range(n) for j in range(i + 1, n)]

    # Count pairs
    unordered_counts = {p: 0 for p in unordered_pairs}
    ordered_counts = {p: 0 for p in ordered_pairs}

    for mat in rectangles:
        k = len(mat[0])
        for row in mat:
            # Unordered: all combos
            """for i, j in itertools.combinations(range(k), 2):
            up = tuple(sorted((row[i], row[j])))
            unordered_counts[up] += 1"""
            for i in range(k - 1):
                up = tuple(sorted((row[i], row[i + 1])))
                unordered_counts[up] += 1
            # Ordered: only adjacent
            for i in range(k - 1):
                op = (row[i], row[i + 1])
                ordered_counts[op] += 1

    # Plot
    plt.figure(figsize=(16, 5))

    # Unordered
    plt.subplot(1, 2, 1)
    pairs_labels = [f"{a},{b}" for a, b in unordered_pairs]
    counts = [unordered_counts[p] for p in unordered_pairs]
    plt.bar(pairs_labels, counts)
    plt.title("Unordered pairs (absolute counts)")
    plt.xticks(rotation=90)
    plt.ylabel("Count")

    # Ordered
    plt.subplot(1, 2, 2)
    pairs_labels = [f"{a}->{b}" for a, b in ordered_pairs]
    counts = [ordered_counts[p] for p in ordered_pairs]
    plt.bar(pairs_labels, counts)
    plt.title("Ordered (adjacent) pairs (absolute counts)")
    plt.xticks(rotation=90)
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
    return pairs_labels, counts


def replace_functions(expression, functions):
    # This will be used to track the index of function replacements
    function_index = 0

    def recursive_replace(s):
        nonlocal function_index

        # To store the result expression
        result = []
        i = 0

        while i < len(s):
            if s[i : i + 2] == "f(":
                # Find the corresponding closing parenthesis
                open_count = 1  # Number of opening parentheses encountered
                j = i + 2
                while j < len(s) and open_count > 0:
                    if s[j] == "(":
                        open_count += 1
                    elif s[j] == ")":
                        open_count -= 1
                    j += 1

                # Extract the inner expression
                inner_expression = s[i + 2 : j - 1]

                # Apply function replacement based on current index
                f_string = functions[function_index]
                function_index += 1
                replaced_inner = recursive_replace(inner_expression)

                if f_string == "power_two":
                    result.append(f"({replaced_inner})**2")
                elif f_string == "power_three":
                    result.append(f"({replaced_inner})**3")
                elif f_string == "exp":
                    result.append(f"exp({replaced_inner})")
                elif f_string == "ln":
                    result.append(f"ln({replaced_inner})")
                elif f_string == "reciprocal":
                    result.append(f"1/({replaced_inner})")
                elif f_string == "sin":
                    result.append(f"sin({replaced_inner})")

                # Advance the current position past the closing parenthesis
                i = j
            else:
                # Collect normal characters directly
                result.append(s[i])
                i += 1

        return "".join(result)

    return recursive_replace(expression)


def replace_xs(expression, n_v, seed=0):
    random.seed(seed)
    x_vars = [f"x_{i+1}" for i in range(n_v)]

    def recursive_replace(s):
        # Split the current level phrases in the expression
        parts = []
        bracket_count = 0
        last_cut = 0
        for i, c in enumerate(s):
            if c == "(":
                if bracket_count == 0:
                    parts.append(s[last_cut:i])
                    last_cut = i + 1
                bracket_count += 1
            elif c == ")":
                bracket_count -= 1
                if bracket_count == 0:
                    inner_expression, inner_used = recursive_replace(s[last_cut:i])
                    parts.append(f"({inner_expression})")
                    last_cut = i + 1

        if last_cut < len(s):
            parts.append(s[last_cut:])

        # Shuffle a fresh copy of x_vars for the current context
        fresh_x_vars = x_vars[:]
        random.shuffle(fresh_x_vars)

        # Replace 'x' with available fresh_x_vars in the current context
        replaced_parts = []
        used_xs = []

        for part in parts:
            if part:
                replacements = []
                for item in part.split("+"):
                    item_clean = item.strip()
                    if item_clean == "x":
                        if fresh_x_vars:
                            replacement = fresh_x_vars.pop()
                            replacements.append(replacement)
                            used_xs.append(replacement)
                        else:
                            replacements.append("x")  # If running out of x_vars
                    else:
                        replacements.append(item_clean)
                replaced_parts.append(" + ".join(replacements))

        return "".join(replaced_parts), used_xs

    # Recursive process
    final_expression, used_xs = recursive_replace(expression)

    # Check if any x_vars were unused, attempt further replacements if possible
    unused_xs = list(set(x_vars) - set(used_xs))

    # Replace any standalone 'x' left with unused x_vars
    # should be unnecessary
    expression_parts = final_expression.split("+")
    final_replacements = []
    for part in expression_parts:
        stripped_part = part.strip()
        if stripped_part == "x" and unused_xs:
            final_replacements.append(unused_xs.pop())
        else:
            final_replacements.append(stripped_part)

    exp = " + ".join(final_replacements)

    var_re = re.compile(r"x_\d+")

    # 1) find which vars are already used
    found = var_re.findall(exp)
    used = set(found)
    unused = list(set(x_vars) - used)

    # 2) for each missing var, pick a donor and overwrite one occurrence
    for x_u in unused:
        # count current occurrences
        counts = {}
        for var in var_re.findall(exp):
            counts[var] = counts.get(var, 0) + 1

        # pick a donor that occurs more than once
        donors = [var for var, c in counts.items() if c > 1]
        if not donors:
            raise RuntimeError("Ran out of donors but still have missing vars!")

        x_d = random.choice(donors)

        # locate all occurrences of x_d
        matches = [m for m in var_re.finditer(exp) if m.group() == x_d]
        choice = random.choice(matches)
        start, end = choice.span()

        # splice in x_u
        exp = exp[:start] + x_u + exp[end:]

    return exp


def sample_structure(
    n=3,
    n_v=2,
    binary_probs=(1 / 2, 1 / 2),
    ternary_probs=(1 / 3, 1 / 3, 1 / 3),
    seed=0,
    x_logit=1,
    f_logit=1,
    p_logit=1,
):
    """
    Sample from the grammar:
    S   ::= +(C0, C0) | f1(T1)
    T1  ::= x | f2(T2) | +(C1, C1)
    T2  ::= x | +(C2, C2)
    T0  ::= x | f1(T1) | +(C0, C0)

    With additional constraints:
        - exactly n functions
        - no more than n_v variables directly inside a function or the overal expression
        - at least n_v variables in total

    And sampling probabily distributions:
        - binary_probs for x|f or x|+ cases (due to constraints)
        - ternary_probs for x|f|+ cases
    """

    def s(n=3, n_v=2):
        """
        n := number of unary functions
        n_v := number of different inputs
        leaf := amount of leaves that are left, decreases per each branching
        """
        if n == 0:
            return " + ".join(["x" for _ in range(n_v)])

        result = None

        while result is None:
            # left over available total leaves (bullshit, enforcing local rule, only, entails global rule in this case)
            leaves = n_v * n
            # left over available leaves per function
            local_leaves = n_v - 1
            case = random.random()
            binary_probs = np.array([f_logit, p_logit]) / (f_logit + p_logit)
            if case < binary_probs[0]:
                n_, leaves_, result, _ = t1(n - 1, leaves, n_v=n_v)
                result = f"f({result})"
                if n_ == 0 and (result.count("x") >= n_v):
                    return result
                else:
                    result = None
            else:
                n_left, leaves_left, result_left, local_leaves = t0(
                    n, leaves - 1, local_leaves - 1, n_v=n_v
                )
                # print(f"t0 left yields: {result_left}")
                n_right, leaves_right, result_right, local_leaves = t0(
                    n_left, leaves_left, local_leaves=local_leaves, n_v=n_v
                )
                # print(f"t0 right yields: {result_right}")
                if n_right == 0 and ((result_left + result_right).count("x") >= n_v):
                    return f"{result_left} + {result_right}"
                else:
                    result = None

    def t0(n, leaves, local_leaves=None, n_v=n_v):
        if local_leaves is None:
            local_leaves = n_v - 1
        case = random.random()
        # print(f"entering t0 with local_leaves={local_leaves} and leaves={leaves}")
        if (n > 0) and (leaves > 0) and (local_leaves > 0):
            ternary_probs = np.array([x_logit, f_logit, p_logit]) / (
                x_logit + f_logit + p_logit
            )
            if case < ternary_probs[0]:
                return n, leaves, "x", local_leaves
            elif case < sum(ternary_probs[:2]):
                n, leaves, result, _ = t1(n - 1, leaves, n_v=n_v)
                result = f"f({result})"
                return n, leaves, result, local_leaves + 1
            else:
                n_left, leaves_left, result_left, local_leaves = t0(
                    n, leaves - 1, local_leaves - 1, n_v=n_v
                )
                n_right, leaves_right, result_right, local_leaves = t0(
                    n_left, leaves_left, local_leaves, n_v=n_v
                )
                return (
                    n_right,
                    leaves_right,
                    f"{result_left} + {result_right}",
                    local_leaves,
                )

        elif n > 0:
            binary_probs = np.array([x_logit, f_logit]) / (x_logit + f_logit)
            if case < binary_probs[0]:
                return n, leaves, "x", local_leaves
            else:
                n, leaves, result, _ = t1(n - 1, leaves, n_v=n_v)
                result = f"f({result})"
                return n, leaves, result, local_leaves + 1

        elif leaves > 0 and (local_leaves > 0):
            binary_probs = np.array([x_logit, p_logit]) / (x_logit + p_logit)
            if case < binary_probs[0]:
                return n, leaves, "x", local_leaves
            else:
                n_left, leaves_left, result_left, local_leaves = t0(
                    n, leaves - 1, local_leaves - 1, n_v=n_v
                )
                n_right, leaves_right, result_right, local_leaves = t0(
                    n_left, leaves_left, local_leaves, n_v=n_v
                )
                return (
                    n_right,
                    leaves_right,
                    f"{result_left} + {result_right}",
                    local_leaves,
                )
        else:
            return n, leaves, "x", local_leaves

    def t1(n, leaves, local_leaves=None, n_v=n_v):
        if local_leaves is None:
            local_leaves = n_v - 1

        case = random.random()

        if (n > 0) and (leaves > 0) and (local_leaves > 0):
            ternary_probs = np.array([x_logit, f_logit, p_logit]) / (
                x_logit + f_logit + p_logit
            )
            if case < ternary_probs[0]:
                return n, leaves, "x", local_leaves
            elif case < sum(ternary_probs[:2]):
                n, leaves, result, _ = t2(n - 1, leaves, n_v=n_v)
                result = f"f({result})"
                return n, leaves, result, local_leaves + 1
            else:
                n_left, leaves_left, result_left, local_leaves = t1(
                    n, leaves - 1, local_leaves - 1, n_v=n_v
                )
                n_right, leaves_right, result_right, local_leaves = t1(
                    n_left, leaves_left, local_leaves, n_v=n_v
                )
                return (
                    n_right,
                    leaves_right,
                    f"{result_left} + {result_right}",
                    local_leaves,
                )

        elif n > 0:
            binary_probs = np.array([x_logit, f_logit]) / (x_logit + f_logit)
            if case < binary_probs[0]:
                return n, leaves, "x", local_leaves
            else:
                n, leaves, result, _ = t2(n - 1, leaves, n_v=n_v)
                result = f"f({result})"
                return n, leaves, result, local_leaves + 1

        elif leaves > 0 and (local_leaves > 0):
            binary_probs = np.array([x_logit, p_logit]) / (x_logit + p_logit)
            if case < binary_probs[0]:
                return n, leaves, "x", local_leaves
            else:
                n_left, leaves_left, result_left, local_leaves = t1(
                    n, leaves - 1, local_leaves - 1, n_v=n_v
                )
                n_right, leaves_right, result_right, local_leaves = t1(
                    n_left, leaves_left, local_leaves, n_v=n_v
                )
                return (
                    n_right,
                    leaves_right,
                    f"{result_left} + {result_right}",
                    local_leaves,
                )
        else:
            return n, leaves, "x", local_leaves

    def t2(n, leaves, local_leaves=None, n_v=n_v):
        if local_leaves is None:
            local_leaves = n_v - 1

        case = random.random()

        if leaves > 0 and (local_leaves > 0):
            binary_probs = np.array([x_logit, p_logit]) / (x_logit + p_logit)
            if case < binary_probs[0]:
                return n, leaves, "x", local_leaves
            else:
                n_left, leaves_left, result_left, local_leaves = t2(
                    n, leaves - 1, local_leaves - 1, n_v=n_v
                )
                n_right, leaves_right, result_right, local_leaves = t2(
                    n_left, leaves_left, local_leaves, n_v=n_v
                )
                return (
                    n_right,
                    leaves_right,
                    f"{result_left} + {result_right}",
                    local_leaves,
                )
        else:
            return n, leaves, "x", local_leaves

    random.seed(seed)
    return s(n=n, n_v=n_v)


def find_k4(pairs):
    n_rows = 4
    n_cols = 4
    n = 6
    symbols = set(range(n))
    # There are 3 adjacent pairs per row, so slots are (row, pos) with pos in 0,1,2
    slots = [(r, c) for r in range(n_rows) for c in range(n_cols - 1)]
    assert len(pairs) == len(slots)

    # Try all possible assignments of pairs to adjacent slots
    for ps in tqdm(itertools.permutations(pairs)):
        # Build an empty Latin rectangle for 4 rows, 4 columns
        mat = [[None] * n_cols for _ in range(n_rows)]
        ok = True
        # Assign all pairs to slots
        for idx, (row, col) in enumerate(slots):
            a, b = ps[idx]
            if mat[row][col] is not None and mat[row][col] != a:
                ok = False
                break
            if mat[row][col + 1] is not None and mat[row][col + 1] != b:
                ok = False
                break
            mat[row][col] = a
            mat[row][col + 1] = b
        if not ok:
            continue
        # Fill the single missing value in each row and col to complete permutation
        for r in range(n_rows):
            row_syms = set(mat[r])
            if None in row_syms:
                need = symbols - set(x for x in mat[r] if x is not None)
                idxs = [i for i in range(n_cols) if mat[r][i] is None]
                if len(need) == len(idxs) == 1:
                    mat[r][idxs[0]] = need.pop()
                else:
                    ok = False
                    break
        # Check columns
        for c in range(n_cols):
            col = [mat[r][c] for r in range(n_rows)]
            if len(set(col)) != n_rows:
                ok = False
                break
        # Check rows
        for r in range(n_rows):
            if len(set(mat[r])) != n_cols:
                ok = False
                break
        if ok:
            print("Found a solution for the first four rows:")
            return mat
    print("No such partial latin rectangle found.")
    return False


def find_k2(k3):
    # Only forbid (row[0], row[1]) and (row[1], row[2]) for each row in k3
    forbidden_k2_pairs = set((row[i], row[i + 1]) for row in k3 for i in (0, 1))

    required_unordered = {tuple(sorted(x)) for x in [[0, 3], [1, 4], [2, 5]]}

    n = 6
    symbols = list(range(n))

    # All possible candidate rows (a,b): a != b, (a,b) not forbidden, no repeats per row/column etc.
    candidates = []
    for a, b in itertools.permutations(symbols, 2):
        if (a, b) in forbidden_k2_pairs:
            continue
        candidates.append((a, b))

    def is_valid_rect(rows):
        if len(rows) != 6:
            return False
        # Latin rectangle property: all 6 symbols in col 0 and col 1
        if set(r[0] for r in rows) != set(symbols):
            return False
        if set(r[1] for r in rows) != set(symbols):
            return False
        # Contains required unordered pairs
        seen_unordered = set(tuple(sorted(r)) for r in rows)
        if not required_unordered.issubset(seen_unordered):
            return False
        return True

    for k2_rows in itertools.permutations(candidates, 6):
        if len(set(k2_rows)) != 6:
            continue
        if is_valid_rect(k2_rows):
            return k2_rows
    else:
        print("No valid k2 found!")


global_seed_gen = itertools.count()


def get_fresh_seed():
    return next(global_seed_gen)


def generate_equations(
    functions=FUNCTIONS,
    operators=OPERATORS,
    problem_domains=PROBLEM_DOMAINS,
    overall_equations_seed=42,
):
    # todo: include constants an vars domains!!!

    problem_groups = []

    seeds = np.genfromtxt("seeds.csv", delimiter=",").astype("int")
    equation_seed_selector = np.random.default_rng(seed=overall_equations_seed)
    equation_seeds = equation_seed_selector.choice(
        seeds, size=problem_domains["n_equations"][0], replace=False
    )
    sampling_seed_selector = np.random.default_rng(seed=overall_equations_seed)
    sampling_seeds = sampling_seed_selector.choice(
        seeds, size=problem_domains["re-samplings"][0], replace=False
    )

    problem_domains["sampling_seed"] = sampling_seeds
    problem_domains["equation_seed"] = equation_seeds

    del problem_domains["n_equations"]
    del problem_domains["re-samplings"]

    problem_grid = hp_meshgrid(problem_domains, problem_groups)

    functions_prior = priors_from_space(list(functions.keys()))
    operators_prior = priors_from_space(list(operators.keys()))
    equations = []

    grid_config = dict()
    grid_config["structures"] = []
    grid_config["functions"] = []
    grid_config["operators"] = []
    grid_config["features"] = []

    structures = [
        [0, 1, 1],
        [0, 1, 2, 2],
        [0, 1, 1, 2],
        [0, 1, 2, 1],
        [0, 1, 2, 1, 2],
        [0, 1, 2, 2, 3],
        [0, 1, 2, 3, 2],
        [0, 1, 2, 2, 1],
        [0, 1, 1, 2, 2],
    ]
    for params in problem_grid:
        features_prior = {"constants": 0.5, "variables": 0.5}
        # structure_prior = structure_prior_from_depth(params["depth"])
        if params["max_nodes"] == 5:
            structure_prior = {
                str(struct): 1 / len(structures[-5:]) for struct in structures[-5:]
            }
        elif params["max_nodes"] == 4:
            structure_prior = {
                str(struct): 1 / len(structures[1:4]) for struct in structures[1:4]
            }
        else:
            structure_prior = {
                str(struct): 1 / len(structures[:1]) for struct in structures[:1]
            }

        prior = {
            "structures": structure_prior,
            "functions": functions_prior,
            "operators": operators_prior,
            "features": features_prior,
        }
        grid_config["structures"].append(structure_prior)
        grid_config["functions"].append(functions_prior)
        grid_config["operators"].append(operators_prior)
        grid_config["features"].append(features_prior)

        np.random.seed(params["equation_seed"])
        # afgfdg

        equation = None
        seed = params["equation_seed"]
        while equation is None:
            equation = equation_tree.sample(
                n=1, prior=prior, max_num_variables=params["max_vars"]
            )[0]
            seed += 1
            np.random.seed(seed)

        equations += [
            (
                equation,
                params["sampling_seed"],
                seed,
                params["training_domain"],
                params["constants_domain"],
            )
        ]
        # print("fresh sample: ", equations[-1])
    # print(equations)

    grid_config["problem_domains"] = problem_domains
    grid_config["problem_groups"] = problem_groups
    return grid_config, equations


'''class TestMonitor(DARTSExecutionMonitor):
    def __init__(self, x_test, y_test):
        """
        Initializes the execution monitor.
        """
        super().__init__()
        self.test_loss_history = list()
        self.test_loss_stepwise_history = list()
        self.x_test = x_test
        self.y_test = y_test
        self.dummy_regressor = DARTSRegressor()

    def train_loss_on_epoch_end(
        self,
        network,
        architect,
        epoch,
        **kwargs,
    ):
        """
        A function to monitor the execution of the DARTS algorithm.

        Arguments:
            network: The DARTS network containing the weights each operation
                in the mixture architecture
            architect: The architect object used to construct the mixture architecture.
            epoch: The current epoch of the training.
            **kwargs: other parameters which may be passed from the DARTS optimizer
        """
        super().execution_monitor(
            network,
            architect,
            epoch,
            **kwargs,
        )
        # collect data for visualization

    def test_loss_on_epoch_end(
        self,
        network,
        **kwargs,
    ):
        model = _generate_model(
            network_=network,
            output_type=kwargs["output_type"],
            sampling_strategy=kwargs["sampling_strategy"],
            data_loader=kwargs["data_loader"],
            param_update_steps=kwargs["param_updates_for_sampled_model"],
            param_learning_rate_max=kwargs["param_learning_rate_max"],
            param_learning_rate_min=kwargs["param_learning_rate_min"],
            param_momentum=kwargs["param_momentum"],
            param_weight_decay=kwargs["param_weight_decay"],
            grad_clip=kwargs["grad_clip"],
        )
        self.dummy_regressor.model_ = model
        self.test_loss_history.append(
            score(self.dummy_regressor, self.x_test, self.y_test)
        )

    def test_loss_on_step(
        self,
        network,
        **kwargs,
    ):
        model = _generate_model(
            network_=network,
            output_type="real",
            sampling_strategy="max",
            data_loader=kwargs["data_loader"],
            param_update_steps=2,
            param_learning_rate_max=kwargs["param_learning_rate_max"],
            param_learning_rate_min=kwargs["param_learning_rate_min"],
            param_momentum=kwargs["param_momentum"],
            param_weight_decay=kwargs["param_weight_decay"],
            grad_clip=kwargs["grad_clip"],
        )
        self.dummy_regressor.model_ = model
        print(score(self.dummy_regressor, self.x_test, self.y_test))
        self.test_loss_stepwise_history.append(
            score(self.dummy_regressor, self.x_test, self.y_test)
        )'''


def score(
    self, x: np.ndarray, y: np.ndarray, metric=sklearn.metrics.mean_squared_error
):
    """
    By capitalizing X, just because it potentially has a dimension more than y, sklearn API commits a crime against nature. Know what, fuck sklearn, I make it lowercase.
    """
    y_pred = self.predict(x)
    return metric(y, y_pred)


def remove_unit_coeff(expr):
    # Mul (multiplications)
    if isinstance(expr, sp.Mul):
        args = [a for a in expr.args if not (a == 1 or a == 1.0)]
        return sp.Mul(*args)
    # recursively process args for Add and Mul
    elif expr.args:
        return expr.func(*[remove_unit_coeff(a) for a in expr.args])
    else:
        return expr


def ensure_symbolic(x):
    if isinstance(x, sp.Basic):
        return x
    else:
        return sp.sympify(x)


def norm_sin(s):
    arg = s.args[0]
    # split arg = var_part + const_part
    const, varpart = arg.as_independent(*arg.free_symbols, as_Add=True)
    # reduce the constant into [0,2π)
    phi = float(const % (2 * float(sp.pi)))
    return sp.sin(varpart + phi)


def unify_float_precision(expr, precision=10):
    # build a mapping old_Float → new_Float(at the desired precision)
    repl = {f: sp.Float(str(f), precision=precision) for f in expr.atoms(sp.Float)}
    return expr.xreplace(repl)


def simplify(exp, decimal_points=3, sensitivity=0.1):  # naive
    """
    Simplifies the input expression by rounding numerical values to a specified number of decimal points
    and applying a sensitivity threshold.

    Parameters:
        exp: The input sympy expression.
        decimal_points (int): Number of decimal points to round numbers to. Default is 3.
        sensitivity (float): Values below this threshold will be set to zero. Default is 0.1.

    Returns:
        The simplified expression with numbers rounded and small values adjusted according to sensitivity.
    """
    for a in sp.preorder_traversal(exp):
        if isinstance(a, sp.Float):
            exp = exp.subs(a, round(a, decimal_points))

    # recursive I guess....
    for a in sp.preorder_traversal(exp):
        if isinstance(a, sp.Float):
            # Round to one decimal
            rounded = round(a, decimal_points)
            # Replace in expression with zero if the value is less than 0.1
            if abs(rounded) < sensitivity:
                exp = exp.subs(a, sp.Float(0))
            else:
                exp = exp.subs(a, rounded)
    exp = ensure_symbolic(exp)
    return remove_unit_coeff(exp)


def str_to_sympy(string, var="x_"):
    """
    Variables are expected to be called x_1, x_2, etc.
    Includes dynamic evaluation of sympy expressions.
    """
    var_names = re.findall(r"{}\d+".format(var), string)
    symbols_dict = {name: sp.symbols(name) for name in var_names}

    # Make all available sympy functions and constants available in eval by combining them with symbols
    all_sympy_names = {
        name: obj
        for name, obj in sp.__dict__.items()  # same as vars(sp)
        if not name.startswith("_")  # skip private helpers
        and not isinstance(obj, types.ModuleType)  # skip sub-modules
    }
    all_context = {**symbols_dict, **all_sympy_names}
    all_context

    # Safe eval with all sympy names
    # print("---------------------")
    # print(string)
    # print(all_context)
    return eval(string, {"__builtins__": {}}, all_context)


# Function to convert NumPy arrays to lists
def convert_to_serializable(data):
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert NumPy array to list
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    else:
        return str(
            data
        )  # Return the data as is if it's neither a dict nor a numpy array


def hp_meshgrid(hp_domains, groups):
    """
    Generates a meshgrid of hyperparameter values based on the specified domains for each key.

    Parameters:
    hp_domains (dict): A dictionary where keys are hyperparameters and values are lists
                       of possible values.
    groups (list of tuples): A list of tuples, where each tuple contains keys that should
                             have their values aligned by index. E.g.
                             groups=[('lr1', 'lr2'),] will make sure, that for each
                             experiment the learning rates at the same index in their
                             respective list in hp_domains will be used. Usually you
                             will not want different learning rates during the same
                             experiment.

    Returns:
    dict_meshgrid (list): A list of dictionaries containing all combinations of the given
                          domains, respecting the specified aligned groups.
    """

    # Store the indices of each group of keys
    aligned_values = []
    other_keys = SortedSet(hp_domains.keys())  # To collect keys not in any group

    # Loop through each group and collect aligned values
    for group in groups:
        # Get the aligned values for the current group
        indices = range(
            len(hp_domains[group[0]])
        )  # Assumes all in the group are the same length
        aligned_group_values = [
            tuple(hp_domains[key][i] for key in group) for i in indices
        ]
        aligned_values.append(aligned_group_values)

        # Remove these keys from other_keys
        other_keys.difference_update(group)

    # Retain the remaining keys and their values for combinations
    other_keys = sorted(list(other_keys))
    other_keys_values = [hp_domains[key] for key in other_keys]

    # Generate all combinations, making sure aligned groups are preserved
    dict_meshgrid = []
    for other_combination in itertools.product(*other_keys_values):
        for aligned_set in itertools.product(*aligned_values):
            # Create the new dictionary
            combination = {
                key: value for key, value in zip(other_keys, other_combination)
            }

            # Add aligned values to the combination
            for index, group in enumerate(groups):
                for key, value in zip(group, aligned_set[index]):
                    combination[key] = value

            dict_meshgrid.append(combination)

    return dict_meshgrid


def save_experiment_config(config, folder="configs"):
    number = str(sum(file.startswith("config") for file in os.listdir(folder))).zfill(4)
    config = convert_to_serializable(config)
    with open(f"{folder}/config_{number}.json", "w") as f:
        json.dump(config, f)


def save_grid_config(config, folder="gridsearch"):
    number = str(
        sum(file.startswith("gridsearch") for file in os.listdir(folder))
    ).zfill(3)
    config = convert_to_serializable(config)
    with open(f"{folder}/gridsearch_{number}.json", "w") as f:
        json.dump(config, f)


def load_config(prefix, number, n_digits=None):
    if n_digits is None:
        n_digits = 4 if "configs" in prefix else 3
    number = str(number).zfill(n_digits)
    with open(f"{prefix}{number}.json", "rt") as f:
        config = json.load(f)
    return config


def load_equations(filename, var="x_"):
    with open(filename, "rt") as f:
        eqs = f.read().splitlines()
    print(eqs)
    # return [equation_tree.EquationTree.from_sympy(str_to_sympy(eq, var=var)) for eq in eqs]
    return [str_to_sympy(eq, var=var) for eq in eqs]


def sympy_to_dataset(eq, n=1000, seed=42, domain=(-1.0, 1.0), sampling="uniform"):
    np.random.seed(seed)
    # Obtain the free variables, sorted as 'x_1', 'x_2', ...
    vars_sorted = [sp.symbols(f"x_{i+1}") for i in range(len(eq.free_symbols))]
    # Uniformly sample 4*n inputs in the domain
    low, high = domain
    sample_count = 4 * n
    X = np.random.uniform(low, high, size=(sample_count, len(vars_sorted)))
    # Make a numpy-callable version of the equation
    fnum = sp.lambdify(vars_sorted, eq, "numpy")
    # Evaluate
    Y = fnum(*[X[:, i] for i in range(len(vars_sorted))])
    Y = np.array(Y)
    # Find non-nan evaluations
    valid_mask = ~np.isnan(Y)
    valid_X = X[valid_mask]
    valid_Y = Y[valid_mask][:, None]
    if valid_X.shape[0] < n:
        raise SamplingError(
            f"Fewer than n ({n}) non-nan samples. Only {valid_X.shape[0]} found for {eq}."
        )
    # Take first n valid samples
    X_out = valid_X[:n]
    Y_out = valid_Y[:n]
    return (
        {
            "train_input": X_out,  # shape (n, n_free_variables)
            "train_label": Y_out,  # shape (n,)
        },
        eq,
    )


def equation_to_dataset(
    equation,
    n=1000,
    seed=42,
    domain=(-1.0, 1.0),
    sampling="uniform",
):
    """
    expects EquationTree
    raises BadDomainException
    todo: check if you gotta get rid of dfs in memory
    """

    print(f"sampling from {equation}\n with seed: {seed}")
    datasets, fixed_equations, running_seed = create_datasets(
        [equation],
        n=n,
        seed=seed,
        domain=domain,
        sampling=sampling,
        constants_domain=(-1.0, 1.0),
    )
    dataset = datasets[0]
    equation = fixed_equations[0].sympy_expr

    var = sorted(list(map(str, equation.free_symbols)))
    print(var)
    # print(f"var: {var}")
    pykan_dataset = dict()  # pd.DataFrame()
    # pykan_dataset=dataset.drop(columns=["y"]).values.tolist()
    # print(pykan_dataset.agg(lambda x: [].append, axis=1))
    pykan_dataset["train_label"] = torch.tensor(dataset["observation"].to_numpy())[
        :, None
    ]

    annoying = dataset.loc[:, var].apply(lambda row: row.tolist(), axis=1).to_numpy()
    indim = len(annoying[0])

    pykan_dataset["train_input"] = torch.tensor(
        np.array([item for row in annoying for item in row]).reshape((n, indim))
    )
    return pykan_dataset


def equation_to_datasets(
    equation,
    n=1000,
    seed=42,
    domain=(-1.0, 1.0),
    sampling="uniform",
    test_domain=None,
    constants_domain=(-1.0, 1.0),
    val_domain=None,
    val_size=None,
):
    """
    expects EquationTree
    raises BadDomainException
    todo: check if you gotta get rid of dfs in memory
    """
    if val_size == None:
        val_size = n // 10
    if test_domain is None:
        test_domain = domain
    if val_domain is None:
        val_domain = [domain, domain]
    print(f"sampling from {equation}\n with seed: {seed}")
    datasets, fixed_equations, running_seed = create_datasets(
        [equation],
        n=n,
        seed=seed,
        domain=domain,
        sampling=sampling,
        constants_domain=constants_domain,
    )
    dataset = datasets[0]
    equation = fixed_equations[0].sympy_expr

    var = sorted(list(map(str, equation.free_symbols)))
    # print(f"var: {var}")
    pykan_dataset = dict()  # pd.DataFrame()
    # pykan_dataset=dataset.drop(columns=["y"]).values.tolist()
    # print(pykan_dataset.agg(lambda x: [].append, axis=1))
    pykan_dataset["train_label"] = torch.tensor(dataset["observation"].to_numpy())[
        :, None
    ]

    annoying = dataset.loc[:, var].apply(lambda row: row.tolist(), axis=1).to_numpy()
    indim = len(annoying[0])

    pykan_dataset["train_input"] = torch.tensor(
        np.array([item for row in annoying for item in row]).reshape((n, indim))
    )

    if test_domain is None:
        testsets, _, running_seed = create_datasets(
            fixed_equations,
            n=n // 10,
            seed=running_seed,
            domain=domain,
            sampling=sampling,
            constants_set=True,
        )
    else:
        distribution = lambda: np.random.uniform(test_domain[0], test_domain[1])
        testsets, _, running_seed = create_datasets(
            fixed_equations,
            n=n,
            seed=running_seed,
            domain=domain,
            sampling=sampling,
            distribution=distribution,
            constants_set=True,
        )

    if val_domain is None:
        valsets, _, running_seed = create_datasets(
            fixed_equations,
            n=val_size,
            seed=running_seed,
            domain=domain,
            sampling=sampling,
            constants_set=True,
        )
    else:
        if np.max(val_domain[0]) > np.min(val_domain[1]):
            distribution = lambda: np.random.uniform(
                np.min(val_domain[0]), np.max(val_domain[1])
            )
        else:
            distribution = lambda: np.random.choice(
                [np.random.uniform(min_val, max_val) for min_val, max_val in val_domain]
            )
        valsets, _, running_seed = create_datasets(
            fixed_equations,
            n=val_size,
            seed=running_seed,
            domain=domain,
            sampling=sampling,
            distribution=distribution,
            constants_set=True,
        )

    valset = valsets[0]
    testset = testsets[0]
    print(f"fixed still fixed: {fixed_equations}, {_}")

    pykan_dataset["test_label"] = torch.tensor(testset["observation"].to_numpy())[
        :, None
    ]

    annoying = testset.loc[:, var].apply(lambda row: row.tolist(), axis=1).to_numpy()
    indim = len(annoying[0])

    pykan_dataset["test_input"] = torch.tensor(
        np.array([item for row in annoying for item in row]).reshape((n, indim))
    )

    pykan_dataset["val_label"] = torch.tensor(valset["observation"].to_numpy())[:, None]

    annoying = valset.loc[:, var].apply(lambda row: row.tolist(), axis=1).to_numpy()
    indim = len(annoying[0])

    pykan_dataset["val_input"] = torch.tensor(
        np.array([item for row in annoying for item in row]).reshape((val_size, indim))
    )

    number = str(sum(file.startswith("df") for file in os.listdir()))
    # pd.DataFrame(pykan_dataset["train_input"]).to_csv(f"df_{number}.csv")

    return equation, pykan_dataset


class BadDomainException(Exception):
    pass


def create_datasets(
    equations,
    n=100,
    seed=42,
    domain=(-1.0, 1.0),
    sampling="uniform",
    constants_domain=(-1.0, 1.0),
    distribution=None,
    constants_set=False,
):
    datasets = []
    fixed_equations = []
    print("running seed:", seed)

    for i, equation in tqdm(enumerate(equations), total=len(equations)):
        constants_fine = False

        if equation is None:  # todo: find out why we get None from sampler
            continue

        j = 0
        while not constants_fine:
            np.random.seed(seed + j)

            if not constants_set:
                constants_prior = lambda: np.random.uniform(
                    constants_domain[0], constants_domain[1], 1
                )[0]

                equation = instantiate_constants(equation, constants_prior)

            print(f"fixed equation: {equation}")
            # equation.create_dataset=create_dataset
            # dataset=equation.create_dataset(equation, 10)
            np.random.seed(seed + j)
            # print(seed+j)
            # print("random state hint before:", np.random.random(5))

            equation.get_evaluation(
                min_val=domain[0],
                max_val=domain[1],
                num_samples=n * 4,
                distribution=distribution,
            )
            dataset = equation.value_samples_as_df

            if dataset.isnull().values.all():
                j += 1
                continue

            # Use the replace method to set inf/-inf to NaN
            dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

            """# Optionally, clip values to the limits of float64 if they are too extreme
            float64_max = np.finfo(np.float64).max
            float64_min = -float64_max
            dataset = dataset.clip(lower=float64_min, upper=float64_max)"""

            try:
                dataset = dataset.dropna().iloc[np.arange(n)].reset_index(drop=True)
                datasets.append(dataset)
                fixed_equations.append(equation)
            except IndexError:
                # todo: implement domain analysis instead
                raise BadDomainException("bad equation:\n" + str(equation))

            break

    # print(f"example data: {datasets[0].head()}")
    return datasets, fixed_equations, seed + j + 1


def default_pykan(equation, seed=42, hidden_size=5):
    device = (
        torch.get_default_device()
    )  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        in_size = len(set(equation.variables))
    except AttributeError:
        in_size = len(equation.free_symbols)

    print([in_size, *hidden_size, 1])
    model = KAN(width=[in_size, *hidden_size, 1], grid=5, k=3, seed=seed, device=device)
    return model


def default_structure(equation, hidden_size=5):

    try:
        in_size = len(set(equation.variables))
    except AttributeError:
        in_size = len(equation.free_symbols)

    return [in_size, *hidden_size, 1]


def testseed():
    print(np.random.random(size=10))


def kan_from_mask(structure, model=None, seed=42, **kwargs):

    structure = copy.deepcopy(structure)

    device = torch.get_default_device()

    for i in range(len(structure)):
        structure[i] = torch.tensor(structure[i]).float()
    if model is None:
        # mask shape = (in_nodes, out_nodes)
        model = KAN(
            width=[structure[0].shape[0]]
            + [structure[i].shape[1] for i in range(len(structure))],
            grid=5,
            k=3,
            seed=seed,
            device=device,
            **kwargs,
        )

    for m in range(len(model.act_fun)):
        model.act_fun[m].mask.data = structure[m]

    return model


# Context to suppress stdout
class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout  # Save the current stdout
        sys.stdout = open(os.devnull, "w")  # Redirect stdout to null device

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()  # Close the devnull file
        sys.stdout = self._original_stdout  # Restore original stdout


def size_mapping(darts_size, n_inputs):
    # really naive size mapping darts -> layered

    n_darts = lambda size, n_in: sum([n_in + (i) for i in range(size)])
    n_layered = (
        lambda size, n_in: n_in * (2 * n_in)
        + 1
        + sum([2 * n_in + 1 + i for i in range(int(size) - 1)])
    )
    n_params = n_darts(darts_size, n_inputs)
    layered_params = [
        n_layered(layered_size, n_inputs) for layered_size in np.arange(15)
    ]
    layered_size = np.searchsorted(np.array(layered_params), n_params)
    return layered_size


"""
        
from dataclasses import dataclass, field
print(dataclass.__module__)
import dataclasses
print(dataclasses.__file__)
print(field.__module__)

@dataclass
class Monitor:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    test_loss: list = field(default_factory=list)
    alphas: list = field(default_factory=list)"""


class NumericTensorJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that understands NumPy scalars/arrays and (optionally) PyTorch
    tensors.  Complex numbers are *not* written as numbers; they are converted
    to their string representation instead (e.g. "1+2j").
    """

    def default(self, obj):
        # -------- Python & NumPy scalars -----------------------------------
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()  # plain int / float / bool
        if isinstance(obj, (complex, np.complexfloating)):
            return str(obj)  # "1+2j", "-0.5j", …

        # -------- NumPy arrays ---------------------------------------------
        if isinstance(obj, np.ndarray):
            if np.iscomplexobj(obj):
                return obj.astype(str).tolist()  # list of strings
            return obj.tolist()  # list of numbers / bools

        # -------- PyTorch tensors ------------------------------------------
        if torch is not None and isinstance(obj, torch.Tensor):
            t = obj.detach().cpu()
            if t.is_complex():
                return t.numpy().astype(str).tolist()
            return t.tolist()

        # -------- Fallback: use the base class -----------------------------
        return super().default(obj)

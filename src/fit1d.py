from itertools import permutations
import traceback
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
from scipy.optimize import minimize, least_squares
import sympy
import torch

N_RESTARTS = 5

SOLVER_CHOICE = {
    ("sin", "exp"): "lbfgs",
    ("sin", "log"): "nlls",
    ("sin", "square"): "nlls",
    ("sin", "cube"): "lbfgs",
    ("sin", "recip"): "nlls",
    ("exp", "log"): "nlls",
    ("exp", "square"): "lbfgs",
    ("exp", "cube"): "nlls",
    ("exp", "recip"): "nlls",
    ("exp", "sin"): "nlls",
    ("log", "exp"): "nlls",
    ("log", "square"): "lbfgs",
    ("log", "cube"): "nlls",
    ("log", "recip"): "lbfgs",
    ("log", "sin"): "nlls",
    ("square", "exp"): "nlls",
    ("square", "log"): "lbfgs",
    ("square", "cube"): "nlls",
    ("square", "recip"): "lbfgs",
    ("square", "sin"): "nlls",
    ("cube", "exp"): "nlls",
    ("cube", "log"): "lbfgs",
    ("cube", "square"): "nlls",
    ("cube", "recip"): "nlls",
    ("cube", "sin"): "lbfgs",
    ("recip", "exp"): "lbfgs",
    ("recip", "log"): "lbfgs",
    ("recip", "square"): "nlls",
    ("recip", "cube"): "nlls",
    ("recip", "sin"): "nlls",
}

# singularity protection functions
f_inv = lambda x, y_th: (
    (x_th := 1 / y_th),
    y_th / x_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / x) * (torch.abs(x) >= x_th),
)
f_inv2 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 2)),
    y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x**2) * (torch.abs(x) >= x_th),
)
f_inv3 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 3)),
    y_th / x_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / x**3) * (torch.abs(x) >= x_th),
)
f_inv4 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 4)),
    y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x**4) * (torch.abs(x) >= x_th),
)
f_inv5 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 5)),
    y_th / x_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / x**5) * (torch.abs(x) >= x_th),
)
f_sqrt = lambda x, y_th: (
    (x_th := 1 / y_th**2),
    x_th / y_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(torch.sqrt(torch.abs(x)) * torch.sign(x))
    * (torch.abs(x) >= x_th),
)
f_power1d5 = lambda x, y_th: torch.abs(x) ** 1.5
f_invsqrt = lambda x, y_th: (
    (x_th := 1 / y_th**2),
    y_th * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / torch.sqrt(torch.abs(x))) * (torch.abs(x) >= x_th),
)
f_log = lambda x, y_th: (
    (x_th := torch.e ** (-y_th)),
    -y_th * (torch.abs(x) < x_th)
    + torch.nan_to_num(torch.log(torch.abs(x))) * (torch.abs(x) >= x_th),
)
f_tan = lambda x, y_th: (
    (clip := x % torch.pi),
    (delta := torch.pi / 2 - torch.arctan(y_th)),
    -y_th / delta * (clip - torch.pi / 2) * (torch.abs(clip - torch.pi / 2) < delta)
    + torch.nan_to_num(torch.tan(clip)) * (torch.abs(clip - torch.pi / 2) >= delta),
)
f_arctanh = lambda x, y_th: (
    (delta := 1 - torch.tanh(y_th) + 1e-4),
    y_th * torch.sign(x) * (torch.abs(x) > 1 - delta)
    + torch.nan_to_num(torch.arctanh(x)) * (torch.abs(x) <= 1 - delta),
)
f_arcsin = lambda x, y_th: (
    (),
    torch.pi / 2 * torch.sign(x) * (torch.abs(x) > 1)
    + torch.nan_to_num(torch.arcsin(x)) * (torch.abs(x) <= 1),
)
f_arccos = lambda x, y_th: (
    (),
    torch.pi / 2 * (1 - torch.sign(x)) * (torch.abs(x) > 1)
    + torch.nan_to_num(torch.arccos(x)) * (torch.abs(x) <= 1),
)
f_exp = lambda x, y_th: (
    (x_th := torch.log(y_th)),
    y_th * (x > x_th) + torch.exp(x) * (x <= x_th),
)

SYMBOLIC_LIB = {
    "x": (lambda x: x, lambda x: x, 1, lambda x, y_th: ((), x)),
    "square": (lambda x: x**2, lambda x: x**2, 2, lambda x, y_th: ((), x**2)),
    "cube": (lambda x: x**3, lambda x: x**3, 3, lambda x, y_th: ((), x**3)),
    "recip": (lambda x: 1 / x, lambda x: 1 / x, 2, f_inv),
    "exp": (lambda x: torch.exp(x), lambda x: sympy.exp(x), 2, f_exp),
    "log": (lambda x: torch.log(x), lambda x: sympy.log(x), 2, f_log),
    "sin": (
        lambda x: torch.sin(x),
        lambda x: sympy.sin(x),
        2.5,
        lambda x, y_th: ((), torch.sin(x)),
    ),
    "0": (
        lambda x: torch.zeros_like(x),
        lambda x: 0 * x,
        0,
        lambda x, y_th: ((), torch.zeros_like(x)),
    ),
}

# left-over from naming convention inconsistencies between KAN and DARTS
NAME_TO_KEY = {
    "sin": "sin",
    "exp": "exp",
    "log": "log",
    "square": "square",
    "cube": "cube",
    "recip": "recip",
}

# Create all possible combinations of primitives and populate them with the original atoms
SYMBOLIC_TUPLE_LIB = {
    (A, B): tuple(  # four fields in original entry
        (SYMBOLIC_LIB[NAME_TO_KEY[A]][k], SYMBOLIC_LIB[NAME_TO_KEY[B]][k])
        for k in range(4)
    )
    for A, B in permutations(NAME_TO_KEY, 2)  # all ordered pairs, A ≠ B
}


# =========================================================================
# 1)  INNER MAP ϕ  (value and derivative)
# =========================================================================
def inner_exp(z, hi=50, lo=-50):
    zc = np.clip(z, lo, hi)
    v = np.exp(zc)
    return v, v


def inner_log(z, tiny=1e-12):
    zc = np.clip(z, tiny, None)
    return np.log(zc), 1 / zc


def inner_square(z):
    return z**2, 2 * z


def inner_cube(z):
    return z**3, 3 * z**2


def inner_recip(z, clip=1e-2, mx=1e5):
    zc = np.clip(z, -mx, mx)
    small = np.abs(zc) < clip
    zc[small] = clip * np.sign(zc[small] + 1e-16)
    return 1 / zc, -1 / zc**2


def phi_sin(z):  # NEW inner map
    return np.sin(z), np.cos(z)


# -------------------------------------------------------------------------
# dictionary of admissible inner maps
# -------------------------------------------------------------------------
INNER_MAPS = {
    "exp": (inner_exp, "exp"),
    "log": (inner_log, "log"),
    "square": (inner_square, "(·)²"),
    "cube": (inner_cube, "(·)³"),
    "recip": (inner_recip, "1/(·)"),
    "sin": (phi_sin, "sin"),
}


def outer_sin(z):
    return np.sin(z)


def outer_exp(z, hi=50, lo=-50):
    return np.exp(np.clip(z, lo, hi))


def outer_log(z, tiny=1e-12):
    return np.log(np.clip(np.abs(z), tiny, None))


def outer_square(z):
    return z**2


def outer_cube(z):
    return z**3


def outer_recip(z, clip=1e-2, mx=1e5):
    zc = np.clip(z, -mx, mx)
    small = np.abs(zc) < clip
    zc[small] = clip * np.sign(zc[small] + 1e-16)
    return 1.0 / zc


OUTER_MAPS = {
    "sin": outer_sin,
    "exp": outer_exp,
    "log": outer_log,
    "square": outer_square,
    "cube": outer_cube,
    "recip": outer_recip,
}


# =========================================================================
# 2)  synthetic-data generator
# =========================================================================
def generate_synthetic_data(
    outer="sin", inner_name="exp", true_params=None, n_points=400, x_min=-2.0, x_max=2.0
):
    """
    Create data for

        y(x) = a · OUTER( b · ϕ(c·x + d) + e ) + f

    where
        OUTER ∈ {sin, exp, log, square, cube, recip}
        ϕ     ∈ the same set  (inner_name)

    Parameters
    ----------
    outer, inner_name : str
        Keys of the dictionaries OUTER_MAPS / INNER_MAPS (see below).
    true_params : iterable length 6  (a,b,c,d,e,f)
        If None, the default (1.2, 0.8, 1.5, −0.2, 0.7, −0.3) is used.
    n_points, x_min, x_max : int / float
        Sampling grid specification.

    Returns
    -------
    dict  …  with keys
        x, y, x_mu, x_sig, x_std, params, outer, inner
    """

    # --------------------  make sure requested maps exist  ----------------
    if outer not in OUTER_MAPS:
        raise ValueError(f"outer map '{outer}' not recognised")
    if inner_name not in INNER_MAPS:
        raise ValueError(f"inner map '{inner_name}' not recognised")

    phi_fun, _ = INNER_MAPS[inner_name]
    outer_fun = OUTER_MAPS[outer]

    # --------------------  parameters ------------------------------------
    if true_params is None:
        true_params = np.random.normal(
            size=6
        )  # np.array([1.2, 0.8, 1.5, -0.2, 0.7, -0.3])
        # print(true_params)
    a0, b0, c0, d0, e0, f0 = true_params

    # --------------------  sample x and build y ---------------------------
    x = np.linspace(x_min, x_max, n_points)
    phi_val, _ = phi_fun(c0 * x + d0)  # ϕ(c₀ x + d₀)
    y = a0 * outer_fun(b0 * phi_val + e0) + f0

    # --------------------  basic statistics ------------------------------
    x_mu = x.mean()
    x_sig = x.std()
    x_std = (x - x_mu) / x_sig

    return dict(
        x=x,
        y=y,
        x_mu=x_mu,
        x_sig=x_sig,
        x_std=x_std,
        params=true_params,
        outer=outer,
        inner=inner_name,
        n=x.size,
    )


# value-only versions of the six admissible outer maps
def _sin(z):
    return np.sin(z)


def _log(z):
    return np.log(np.clip(np.abs(z), 1e-12, None))


def _exp(z):
    return np.exp(np.clip(z, -50, 50))


def _recip(z):
    zc = np.clip(z, -1e5, 1e5)
    return 1 / zc


def _square(z):
    return z**2


def _cube(z):
    return z**3


_1LAYER_FUNS = {
    "sin": _sin,
    "log": _log,
    "exp": _exp,
    "recip": _recip,
    "square": _square,
    "cube": _cube,
}


def generate_synthetic_data_single(
    fun_name: str = "sin",
    true_params: Sequence[float] | None = None,
    n_points: int = 400,
    x_min: float = -2.0,
    x_max: float = 2.0,
):
    """
    Return a dict analogous to the old generator but for
        y = a·f(b·x + c) + d
    """
    if fun_name not in _1LAYER_FUNS:
        raise KeyError(f"unknown outer function '{fun_name}'")

    if true_params is None:
        true_params = np.array([1.2, 0.8, 1.0, -0.3, 0.2])  # a,b,c,d
    a0, b0, c0, d0 = true_params

    x = np.linspace(x_min, x_max, n_points)
    f_val = _1LAYER_FUNS[fun_name](b0 * x + c0)
    y = a0 * f_val + d0

    x_mu, x_sig = x.mean(), x.std()
    x_std = (x - x_mu) / x_sig

    return dict(
        x=x,
        y=y,
        x_mu=x_mu,
        x_sig=x_sig,
        x_std=x_std,
        params=true_params,
        fun_name=fun_name,
        n=n_points,
    )


# =========================================================================
# 4)  helper: re-parameterise  b = b(s)
# =========================================================================
safe_exp = lambda z, hi=700, lo=-700: np.exp(np.clip(z, lo, hi))

b_log = lambda s: (np.sign(s) * np.exp(np.abs(s)), np.exp(np.abs(s)))
b_lin = lambda s: (s, np.ones_like(s))
b_sinh = lambda s: (np.sinh(s), np.cosh(s))

B_MAP = {
    "exp": b_log,
    "exp_solo": b_lin,
    "log": b_lin,
    "square": b_sinh,
    "cube": b_sinh,
    "recip": b_lin,
    "recip_solo": b_log,
    "sin": b_lin,
}


# cubic-specific wrappers
def b_from_s_c(s, inner):
    return B_MAP[inner](s)[0]


def db_ds_c(s, inner):
    return B_MAP[inner](s)[1]


# =========================================================================
# 5)  small linear solver (variable projection)
# =========================================================================
def solve_linear_block(Phi, yvec, ridge_thresh=1e12):
    U, S, VT = svd(Phi, full_matrices=False)
    Sinv = 1 / S
    if S[0] / S[-1] > ridge_thresh:
        lam = 1e-8 * S[0]
        Sinv = S / (S**2 + lam)
    alpha = (VT.T * Sinv) @ (U.T @ yvec)
    return alpha, U


# =========================================================================
# 6)  coordinate conversion + canonicalisation for cubic model
# =========================================================================
def to_raw_coords(theta_std, x_mu, x_sig):
    a, b, c_std, d_std, e, f = theta_std
    c_raw = c_std / x_sig
    d_raw = d_std - c_std * x_mu / x_sig
    return np.array([a, b, c_raw, d_raw, e, f])


def canonicalise_cubic(params, phi_name):
    a, b, c, d, e, f = params
    if phi_name == "square":
        power = 2
    elif phi_name == "cube":
        power = 3
    elif phi_name == "recip":
        power = -1
    else:
        power = None

    if power is not None:
        if c == 0:
            raise ValueError("c must not be zero for maps with scale symmetry")
        s = abs(c)
        if power > 0:
            b *= s**power
        else:
            b /= s
        c = np.sign(c)
        d /= s
    return np.array([a, b, c, d, e, f])


# =========================================================================
# 7)  one fit with cubic outer function
# =========================================================================
def fit_once_cube(seed, x, y, x_mu, x_sig, inner="log", method="lbfgs"):
    rng = np.random.default_rng(seed)
    N = x.size
    x_std = (x - x_mu) / x_sig

    phi_fun, _ = INNER_MAPS[inner]

    # ---------- design matrix Φ(β) with β = (s,c,d,e)
    def build_design(beta):
        s, c, d, e_par = beta
        b = b_from_s_c(s, inner)
        phi, phi_p = phi_fun(c * x_std + d)
        H = b * phi + e_par
        H3 = H**3
        Phi = np.column_stack((H3, np.ones_like(H3)))
        cache = dict(
            H=H, H2=H**2, phi=phi, phi_p=phi_p, b=b, s=s, c=c, d=d, e=e_par, Phi=Phi
        )
        return Phi, cache

    def proj_model(beta):
        Phi, C = build_design(beta)
        alpha, U = solve_linear_block(Phi, y)  # (a,f)
        y_hat = Phi @ alpha
        C.update(alpha=alpha, U=U)
        return y_hat, C

    def residual(beta):
        return (proj_model(beta)[0] - y) / y.std()

    def jacobian(beta):
        y_hat, C = proj_model(beta)
        Phi, (a_hat, _), U = C["Phi"], C["alpha"], C["U"]
        H, H2, phi, phi_p, b = C["H"], C["H2"], C["phi"], C["phi_p"], C["b"]
        orth = lambda v: v - U @ (U.T @ v)

        s, c, d, e_par = beta
        dH_ds = db_ds_c(s, inner) * phi
        dH_dc = b * phi_p * x_std
        dH_dd = b * phi_p
        dH_de = np.ones_like(H)

        col = lambda dH: 3 * a_hat * H2 * dH
        J = np.empty((N, 4))
        J[:, 0] = orth(col(dH_ds))
        J[:, 1] = orth(col(dH_dc))
        J[:, 2] = orth(col(dH_dd))
        J[:, 3] = orth(col(dH_de))
        return J / y.std()

    # ---------- outer optimiser
    if method.lower() == "lbfgs":
        objective = lambda beta: np.mean((proj_model(beta)[0] - y) ** 2) / np.var(y)
        x0 = rng.uniform(-0.5, 0.5, 4)
        bounds = [(-7, 7), (-7, 7), (-5, 5), (-5, 5)]
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 600},
        )
    elif method.lower() == "nlls":
        beta0 = rng.uniform(-0.5, 0.5, 4)
        res = least_squares(
            residual,
            beta0,
            jac=jacobian,
            method="trf",
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
        )
    else:
        raise ValueError("method must be 'lbfgs' or 'nlls'")

    # ---------- assemble parameter vectors
    y_hat, C_fin = proj_model(res.x)
    a_std, f_std = C_fin["alpha"]
    s_hat, c_hat, d_hat, e_hat = res.x
    b_hat = b_from_s_c(s_hat, inner)
    theta_std = np.array([a_std, b_hat, c_hat, d_hat, e_hat, f_std])

    # ---------- errors
    rel_mse_train = np.mean((y_hat - y) ** 2) / np.var(y)

    x_ext = np.linspace(x.min() - 2, x.max() + 2, 800)

    theta_raw = to_raw_coords(theta_std, x_mu, x_sig)
    theta_can = canonicalise_cubic(theta_raw, inner)

    return dict(
        seed=seed,
        method=method.lower(),
        success=res.success,
        message=res.message if hasattr(res, "message") else res.status,
        rel_mse_train=rel_mse_train,
        params_std=theta_std,
        params_raw=theta_raw,
        params_can=theta_can,
        result=res,
        x_ext=x_ext,
        y_hat_train=y_hat,
    )


# Helpers specifically for sin


def b_from_s_sin(s, inner):
    return B_MAP[inner](s)[0]


def db_ds_sin(s, inner):
    return B_MAP[inner](s)[1]


def _flip_b(b, e):
    """
    Change sign of b and adjust the phase so that

          a·sin(b·… + e)   and   a·sin(−b·… + e′)

    describe the same function.
    """
    return -b, (np.pi - e) % (2 * np.pi)


def canonicalise_sin(params, phi_name):
    """
    Bring one parameter vector into a *canonical* form:

        • b ≥ 0                                              (all maps)
        • |c| = 1  if the map admits a continuous rescaling
        • c   = +1 if the sign can be removed without loss

    Continuous rescaling exists for
        square, cube  :  (·)ⁿ          (n even / odd)
        recip         :  1/(·)

    It does *not* exist for
        exp , log     :  cannot absorb |c| into b without
                         changing the functional form.

    Parameters
    ----------
    params   : array_like, shape (6,)
               [a, b, c, d, e, f]   (all in the RAW x–coordinate system!)
    phi_name : str
               'exp' | 'log' | 'square' | 'cube' | 'recip'
    """
    a, b, c, d, e, f = params

    # --------------------------------------------------
    # 1.  enforce  b ≥ 0  (flip sign into phase)
    # --------------------------------------------------
    if b < 0:
        b, e = _flip_b(b, e)

    # --------------------------------------------------
    # 2.  eliminate |c|  where a scale-symmetry exists
    # --------------------------------------------------
    if phi_name == "square":  # (cx+d)²
        power = 2
    elif phi_name == "cube":  # (cx+d)³
        power = 3
    elif phi_name == "recip":  # 1/(cx+d)
        power = -1  # “power” = −1 for convenience
    else:  # 'exp' or 'log'  → no rescaling
        power = None

    if power is not None:
        if c == 0:
            raise ValueError("c must not be zero for maps with scale symmetry")
        s = abs(c)
        if power > 0:  # square / cube
            b *= s**power
        else:  # reciprocal  (power = −1)
            b /= s
        c = np.sign(c)  # now ±1
        d /= s  # d̃ = d / |c|

    # --------------------------------------------------
    # 3.  remove the residual sign of c  (if present)
    # --------------------------------------------------
    if c == -1:
        d = -d
        if phi_name in ("cube", "recip"):  # odd power → extra sign
            e = (np.pi - e) % (2 * np.pi)
        c = +1

    e = e % (2 * np.pi)
    return np.array([a, b, c, d, e, f])


def fit_once_sin(seed, x, y, x_mu, x_sig, inner="log", method="lbfgs"):
    """
    Perform ONE random initialisation of the requested optimisation
    method ∈ {'lbfgs','nlls'} and return a dict with everything that
    is later required for printing / plotting.
    -----------------------------------------------------------------
    Returned keys
        'seed', 'method', 'success', 'message',
        'rel_mse_train', 'rel_mse_ext',
        'params_std', 'params_raw', 'params_can', 'result'
    """
    if inner not in INNER_MAPS:
        raise KeyError(f"inner map '{inner}' not known")

    rng = np.random.default_rng(seed)
    N = x.size
    x_std = (x - x_mu) / x_sig

    # ------------------------------------------------------------------
    # inner map
    # ------------------------------------------------------------------
    phi_fun, _ = INNER_MAPS[inner]

    # --------------- common helpers available in both branches ----------
    def linear_subfit(non_lin_vec):
        """Β = (signed_log_b, c, d)  →  ŷ , θ̂  in std-coords"""
        s, c, d = non_lin_vec.astype(float)
        b = np.sign(s) * safe_exp(np.abs(s))
        phase = b * phi_fun(c * x_std + d)[0]
        Phi = np.column_stack((np.sin(phase), np.cos(phase), np.ones_like(phase)))
        alpha, _ = solve_linear_block(Phi, y)
        A, B, f_std = alpha
        a_std = np.hypot(A, B)
        e_std = np.arctan2(B, A) % (2 * np.pi)
        theta_std = np.array([a_std, b, c, d, e_std, f_std])
        y_hat = Phi @ alpha
        return y_hat, theta_std

    # ===================================================================
    # 6-A)  L-BFGS-B  ---------------------------------------------------
    # ===================================================================
    if method.lower() == "lbfgs":

        def objective(non_lin):
            return np.mean((linear_subfit(non_lin)[0] - y) ** 2) / np.var(y)

        x0 = rng.uniform(-1, 1, 3)  # (s,c,d) initial guess
        bounds = [(-7, 7), (-7, 7), (-5, 5)]

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 500},
        )
        y_hat, theta_std = linear_subfit(res.x)
    # ===================================================================
    # 6-B)  NON-LINEAR LEAST-SQUARES WITH JACOBIAN  ----------------------
    # ===================================================================
    elif method.lower() == "nlls":
        # ------------- build Φ, residual and Jacobian  for variable-proj
        def build_design(beta):
            s, c, d = beta
            b = b_from_s_sin(s, inner)
            phi, phi_p = phi_fun(c * x_std + d)
            phase = b * phi
            sin_p = np.sin(phase)
            cos_p = np.cos(phase)
            Phi = np.column_stack((sin_p, cos_p, np.ones_like(sin_p)))
            return Phi, dict(phi=phi, phi_p=phi_p, sin=sin_p, cos=cos_p, b=b)

        def proj_model(beta):
            Phi, cache = build_design(beta)
            alpha, U = solve_linear_block(Phi, y)
            y_hat = Phi @ alpha
            cache.update(Phi=Phi, alpha=alpha, U=U)
            return y_hat, cache

        def residual(beta):
            return (proj_model(beta)[0] - y) / y.std()

        def jacobian(beta):
            y_hat, C = proj_model(beta)
            Phi, (A, B, _), U = C["Phi"], C["alpha"], C["U"]
            sin, cos, phi, phi_p, b = (
                C[k] for k in ("sin", "cos", "phi", "phi_p", "b")
            )
            # orthogonal complement  (I - P)
            orth = lambda v: v - U @ (U.T @ v)

            s, c, d = beta
            dphi_ds = db_ds_sin(s, inner) * phi
            dphi_dc = b * phi_p * x_std
            dphi_dd = b * phi_p

            col = lambda dphi: A * cos * dphi - B * sin * dphi
            J = np.empty((x.size, 3))
            J[:, 0] = orth(col(dphi_ds))
            J[:, 1] = orth(col(dphi_dc))
            J[:, 2] = orth(col(dphi_dd))
            return J / y.std()

        beta0 = rng.uniform(-0.5, 0.5, 3)  # (s,c,d)
        res = least_squares(
            residual,
            beta0,
            jac=jacobian,
            method="trf",
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
        )

        # transform into "std-parameter-vector" compatible with the other
        # branch ---------------------------------------------------------
        s_hat, c_hat, d_hat = res.x
        b_hat = b_from_s_sin(s_hat, inner)
        phi_fit, _ = phi_fun(c_hat * x_std + d_hat)
        phase_fit = b_hat * phi_fit
        Phi_fit = np.column_stack(
            (np.sin(phase_fit), np.cos(phase_fit), np.ones_like(phase_fit))
        )
        alpha_vec, _ = solve_linear_block(Phi_fit, y)
        A, B, f_std = alpha_vec
        a_std = np.hypot(A, B)
        e_std = np.arctan2(B, A) % (2 * np.pi)
        theta_std = np.array([a_std, b_hat, c_hat, d_hat, e_std, f_std])
        y_hat = Phi_fit @ alpha_vec
    else:
        raise ValueError("method must be 'lbfgs' or 'nlls'")

    # -------------------------------------------------------------------
    #  assemble output statistics common to both solvers
    # -------------------------------------------------------------------
    rel_mse_train = np.mean((y_hat - y) ** 2) / np.var(y)

    # map parameters to raw x domain and canonicalise
    theta_raw = to_raw_coords(theta_std, x_mu, x_sig)
    theta_can = canonicalise_sin(theta_raw, inner)

    return dict(
        seed=seed,
        method=method.lower(),
        success=res.success,
        message=res.message if hasattr(res, "message") else res.status,
        rel_mse_train=rel_mse_train,
        params_std=theta_std,
        params_raw=theta_raw,
        params_can=theta_can,
        result=res,
        y_hat_train=y_hat,
    )


# =========================================================================
# 7′)  one fit with *logarithm* as outer function
# =========================================================================
def fit_once_log(seed, x, y, x_mu, x_sig, inner="exp", method="lbfgs", tiny=1e-12):
    """
    Fit   y = a·log( b·ϕ(c·x + d) + e ) + f   via variable projection.

    Parameters
    ----------
    seed          : int
    x, y          : data
    x_mu, x_sig   : stats of x  (for the derivative wrt c)
    true_params   : ground-truth  [a,b,c,d,e,f]   (only used for plotting)
    inner         : str
        key of INNER_MAPS used for ϕ; must *not* be 'log' itself.
    method        : 'lbfgs' | 'nlls'
    tiny          : float
        safety constant to keep log argument away from 0.

    Returns
    -------
    dict – identical layout to fit_once_cube / fit_once_sin
    """
    if inner not in INNER_MAPS:
        raise KeyError(f"inner map '{inner}' not known")

    rng = np.random.default_rng(seed)
    N = x.size
    x_std = (x - x_mu) / x_sig

    # ------------------------------------------------------------------
    # inner map
    # ------------------------------------------------------------------
    phi_fun, _ = INNER_MAPS[inner]

    # ------------------------------------------------------------------
    # design matrix Φ(β) ,  β = (s, c, d, e)
    # ------------------------------------------------------------------
    def build_design(beta):
        s, c, d, e_par = beta
        b = b_from_s_c(s, inner)
        phi, phi_p = phi_fun(c * x_std + d)
        H = b * phi + e_par  # argument of log
        H_clip = np.clip(np.abs(H), tiny, None)  # avoid log(0)
        logH = np.log(H_clip)
        Phi = np.column_stack((logH, np.ones_like(logH)))

        # quantities reused in Jacobian
        cache = dict(
            H=H,
            phi=phi,
            phi_p=phi_p,
            H_clip=H_clip,
            b=b,
            s=s,
            c=c,
            d=d,
            e=e_par,
            Phi=Phi,
        )
        return Phi, cache

    # ------------------------------------------------------------------
    # projected model  ŷ(β)
    # ------------------------------------------------------------------
    def proj_model(beta):
        Phi, C = build_design(beta)
        alpha, U = solve_linear_block(Phi, y)  # (a,f)
        y_hat = Phi @ alpha
        C.update(alpha=alpha, U=U)
        return y_hat, C

    # residual ----------------------------------------------------------
    def residual(beta):
        return (proj_model(beta)[0] - y) / y.std()

    # analytic Jacobian -------------------------------------------------
    def jacobian(beta):
        y_hat, C = proj_model(beta)
        Phi, (a_hat, _), U = C["Phi"], C["alpha"], C["U"]
        H, H_clip, phi, phi_p, b = (C[k] for k in ("H", "H_clip", "phi", "phi_p", "b"))
        # d log|H| / dH  (with sign(H))
        dlog_dH = np.sign(H) / H_clip

        # derivatives of H wrt β
        s, c, d, e_par = beta
        dH_ds = db_ds_c(s, inner) * phi
        dH_dc = b * phi_p * x_std
        dH_dd = b * phi_p
        dH_de = np.ones_like(H)

        # column builder
        col = lambda dH: a_hat * dlog_dH * dH

        orth = lambda v: v - U @ (U.T @ v)

        J = np.empty((N, 4))
        J[:, 0] = orth(col(dH_ds))
        J[:, 1] = orth(col(dH_dc))
        J[:, 2] = orth(col(dH_dd))
        J[:, 3] = orth(col(dH_de))
        return J / y.std()

    # ------------------------------------------------------------------
    # outer optimiser
    # ------------------------------------------------------------------
    if method.lower() == "lbfgs":
        objective = lambda b: np.mean((proj_model(b)[0] - y) ** 2) / np.var(y)
        x0 = rng.uniform(-0.5, 0.5, 4)
        bounds = [(-7, 7), (-7, 7), (-5, 5), (-5, 5)]
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 600},
        )
    elif method.lower() == "nlls":
        beta0 = rng.uniform(-0.5, 0.5, 4)
        res = least_squares(
            residual,
            beta0,
            jac=jacobian,
            method="trf",
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
        )
    else:
        raise ValueError("method must be 'lbfgs' or 'nlls'")

    # ------------------------------------------------------------------
    # assemble parameter vectors
    # ------------------------------------------------------------------
    y_hat, C_fin = proj_model(res.x)
    a_std, f_std = C_fin["alpha"]
    s_hat, c_hat, d_hat, e_hat = res.x
    b_hat = b_from_s_c(s_hat, inner)
    theta_std = np.array([a_std, b_hat, c_hat, d_hat, e_hat, f_std])

    # ------------------------------------------------------------------
    # errors & canonicalisation
    # ------------------------------------------------------------------
    rel_mse_train = np.mean((y_hat - y) ** 2) / np.var(y)

    theta_raw = to_raw_coords(theta_std, x_mu, x_sig)
    # no special canonicaliser for log – reuse cubic version (only |c|-rescaling)
    theta_can = canonicalise_cubic(theta_raw, inner)

    return dict(
        seed=seed,
        method=method.lower(),
        success=res.success,
        message=res.message if hasattr(res, "message") else res.status,
        rel_mse_train=rel_mse_train,
        params_std=theta_std,
        params_raw=theta_raw,
        params_can=theta_can,
        result=res,
        y_hat_train=y_hat,
    )


# =========================================================================
# 7″)  one fit with *reciprocal* as outer function
# =========================================================================
def fit_once_recip(seed, x, y, x_mu, x_sig, inner="exp", method="lbfgs", clip_val=1e-4):
    """
    Fit   y = a / ( b·ϕ(c·x + d) + e ) + f   via variable projection.
    ----------------------------------------------------------------
    Parameters
    ----------
    seed          : int           random initialisation
    x, y          : ndarray       data
    x_mu, x_sig   : float         stats for x-standardisation
    true_params   : sequence(6)   [a,b,c,d,e,f] — only for plotting
    inner         : str           inner map key (INNER_MAPS)
    method        : 'lbfgs' | 'nlls'
    clip_val      : float         keeps the denominator away from 0
    """
    if inner not in INNER_MAPS:
        raise KeyError(f"inner map '{inner}' not recognised")

    rng = np.random.default_rng(seed)
    N = x.size
    x_std = (x - x_mu) / x_sig

    # ---------- inner map ----------
    phi_fun, _ = INNER_MAPS[inner]

    # ---------- design matrix Φ(β) with β = (s,c,d,e) ----------
    def build_design(beta):
        s, c, d, e_par = beta
        b = b_from_s_c(s, inner)
        phi, phi_p = phi_fun(c * x_std + d)
        H = b * phi + e_par  # denominator
        H_clip = np.where(np.abs(H) < clip_val, clip_val * np.sign(H + 1e-16), H)
        invH = 1.0 / H_clip
        Phi = np.column_stack((invH, np.ones_like(invH)))

        cache = dict(
            H=H_clip,
            H2=invH**2,  # H² in denominator later
            phi=phi,
            phi_p=phi_p,
            b=b,
            s=s,
            c=c,
            d=d,
            e=e_par,
            Phi=Phi,
        )
        return Phi, cache

    # ---------- variable projection ----------
    def proj_model(beta):
        Phi, C = build_design(beta)
        alpha, U = solve_linear_block(Phi, y)  # (a, f)
        y_hat = Phi @ alpha
        C.update(alpha=alpha, U=U)
        return y_hat, C

    resid = lambda beta: (proj_model(beta)[0] - y) / y.std()

    # ---------- analytic Jacobian ----------
    def jacobian(beta):
        y_hat, C = proj_model(beta)
        Phi, (a_hat, _), U = C["Phi"], C["alpha"], C["U"]
        H, H2, phi, phi_p, b = (C["H"], C["H2"], C["phi"], C["phi_p"], C["b"])
        #   d(1/H)/dH = −1/H²
        s, c, d, e_par = beta
        dH_ds = db_ds_c(s, inner) * phi
        dH_dc = b * phi_p * x_std
        dH_dd = b * phi_p
        dH_de = np.ones_like(H)

        col = lambda dH: -a_hat * dH / H**2  # derivative of a·1/H
        orth = lambda v: v - U @ (U.T @ v)

        J = np.empty((N, 4))
        J[:, 0] = orth(col(dH_ds))
        J[:, 1] = orth(col(dH_dc))
        J[:, 2] = orth(col(dH_dd))
        J[:, 3] = orth(col(dH_de))
        return J / y.std()

    # ---------- outer optimiser ----------
    if method.lower() == "lbfgs":
        objective = lambda b: np.mean((proj_model(b)[0] - y) ** 2) / np.var(y)
        x0 = rng.uniform(-0.5, 0.5, 4)
        bounds = [(-7, 7), (-7, 7), (-5, 5), (-5, 5)]
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 600},
        )
    elif method.lower() == "nlls":
        beta0 = rng.uniform(-0.5, 0.5, 4)
        res = least_squares(
            resid, beta0, jac=jacobian, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15
        )
    else:
        raise ValueError("method must be 'lbfgs' or 'nlls'")

    # ---------- assemble parameter vectors ----------
    y_hat, C_fin = proj_model(res.x)
    a_std, f_std = C_fin["alpha"]
    s_hat, c_hat, d_hat, e_hat = res.x
    b_hat = b_from_s_c(s_hat, inner)
    theta_std = np.array([a_std, b_hat, c_hat, d_hat, e_hat, f_std])

    # ---------- error metrics ----------
    rel_mse_train = np.mean((y_hat - y) ** 2) / np.var(y)

    theta_raw = to_raw_coords(theta_std, x_mu, x_sig)
    theta_can = canonicalise_cubic(theta_raw, inner)  # |c|-rescaling

    return dict(
        seed=seed,
        method=method.lower(),
        success=res.success,
        message=res.message if hasattr(res, "message") else res.status,
        rel_mse_train=rel_mse_train,
        params_std=theta_std,
        params_raw=theta_raw,
        params_can=theta_can,
        result=res,
        y_hat_train=y_hat,
    )


# =========================================================================
# 7‴)  one fit with *exponential* as outer function
# =========================================================================
def fit_once_exp(
    seed, x, y, x_mu, x_sig, inner="log", method="lbfgs", hi=50.0, lo=-50.0
):
    """
    Fit   y = a · exp( b·ϕ(c·x + d) + e ) + f   via variable projection.

    Parameters
    ----------
    seed          : int
    x, y          : ndarray
    x_mu, x_sig   : float
    true_params   : sequence(6)   (only for plotting)
    inner         : str           inner map key (INNER_MAPS)
    method        : 'lbfgs' | 'nlls'
    hi, lo        : float         clipping range for exp to avoid overflow

    Returns
    -------
    dict   …  same structure as the other fit_once_* functions
    """
    if inner not in INNER_MAPS:
        raise KeyError(f"inner map '{inner}' not recognised")

    rng = np.random.default_rng(seed)
    N = x.size
    x_std = (x - x_mu) / x_sig

    # ----------------  inner map -----------------
    phi_fun, _ = INNER_MAPS[inner]

    # ----------------  design matrix Φ(β)  with β = (s,c,d,e) -------------
    def build_design(beta):
        s, c, d, e_par = beta
        b = b_from_s_c(s, inner)
        phi, phi_p = phi_fun(c * x_std + d)
        H = b * phi + e_par  # exponent
        expH = safe_exp(H, hi=hi, lo=lo)  # avoid overflow
        Phi = np.column_stack((expH, np.ones_like(expH)))

        cache = dict(
            H=H, expH=expH, phi=phi, phi_p=phi_p, b=b, s=s, c=c, d=d, e=e_par, Phi=Phi
        )
        return Phi, cache

    # ----------------  projected model -----------------
    def proj_model(beta):
        Phi, C = build_design(beta)
        alpha, U = solve_linear_block(Phi, y)  # (a,f)
        y_hat = Phi @ alpha
        C.update(alpha=alpha, U=U)
        return y_hat, C

    # residual ------------------------------------------
    resid = lambda beta: (proj_model(beta)[0] - y) / y.std()

    # analytic Jacobian ---------------------------------
    def jacobian(beta):
        y_hat, C = proj_model(beta)
        Phi, (a_hat, _), U = C["Phi"], C["alpha"], C["U"]
        H, expH, phi, phi_p, b = (C["H"], C["expH"], C["phi"], C["phi_p"], C["b"])

        s, c, d, e_par = beta
        dH_ds = db_ds_c(s, inner) * phi
        dH_dc = b * phi_p * x_std
        dH_dd = b * phi_p
        dH_de = np.ones_like(H)

        col = lambda dH: a_hat * expH * dH  # a·exp(H)·dH
        orth = lambda v: v - U @ (U.T @ v)

        J = np.empty((N, 4))
        J[:, 0] = orth(col(dH_ds))
        J[:, 1] = orth(col(dH_dc))
        J[:, 2] = orth(col(dH_dd))
        J[:, 3] = orth(col(dH_de))
        return J / y.std()

    # ----------------  outer optimisation ----------------
    if method.lower() == "lbfgs":
        objective = lambda b: np.mean((proj_model(b)[0] - y) ** 2) / np.var(y)
        x0 = rng.uniform(-0.5, 0.5, 4)
        bounds = [(-7, 7), (-7, 7), (-5, 5), (-5, 5)]
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 600},
        )
    elif method.lower() == "nlls":
        beta0 = rng.uniform(-0.5, 0.5, 4)
        res = least_squares(
            resid, beta0, jac=jacobian, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15
        )
    else:
        raise ValueError("method must be 'lbfgs' or 'nlls'")

    # ----------------  assemble parameter vectors ----------------
    y_hat, C_fin = proj_model(res.x)
    a_std, f_std = C_fin["alpha"]
    s_hat, c_hat, d_hat, e_hat = res.x
    b_hat = b_from_s_c(s_hat, inner)
    theta_std = np.array([a_std, b_hat, c_hat, d_hat, e_hat, f_std])

    # ----------------  error metrics ---------------------
    rel_mse_train = np.mean((y_hat - y) ** 2) / np.var(y)

    theta_raw = to_raw_coords(theta_std, x_mu, x_sig)
    theta_can = canonicalise_cubic(theta_raw, inner)  # |c|-symmetry only

    return dict(
        seed=seed,
        method=method.lower(),
        success=res.success,
        message=res.message if hasattr(res, "message") else res.status,
        rel_mse_train=rel_mse_train,
        params_std=theta_std,
        params_raw=theta_raw,
        params_can=theta_can,
        result=res,
        y_hat_train=y_hat,
    )


# =========================================================================
# 7⁗)  one fit with *square* as outer function
# =========================================================================
def fit_once_square(seed, x, y, x_mu, x_sig, inner="log", method="lbfgs"):
    """
    Fit  y = a·( b·ϕ(c·x + d) + e )² + f  via variable projection.

    Parameters
    ----------
    seed          : int            random seed for initial guess
    x, y          : ndarray        data
    x_mu, x_sig   : float          stats of x  (for derivative wrt c)
    true_params   : sequence(6)    [a,b,c,d,e,f]  — only for plotting
    inner         : str            key of INNER_MAPS  (ϕ)
    method        : 'lbfgs' | 'nlls'

    Returns
    -------
    dict  –  same structure as fit_once_cube / _sin / _log / _rec / _exp
    """
    if inner not in INNER_MAPS:
        raise KeyError(f"inner map '{inner}' not recognised")

    rng = np.random.default_rng(seed)
    N = x.size
    x_std = (x - x_mu) / x_sig

    # ------------- inner map ϕ -----------------------------------------
    phi_fun, _ = INNER_MAPS[inner]

    # ------------- design matrix Φ(β) with β = (s,c,d,e) ---------------
    def build_design(beta):
        s, c, d, e_par = beta
        b = b_from_s_c(s, inner)
        phi, phi_p = phi_fun(c * x_std + d)
        H = b * phi + e_par  # inner argument
        H2 = H**2
        Phi = np.column_stack((H2, np.ones_like(H2)))
        cache = dict(
            H=H, H2=H2, phi=phi, phi_p=phi_p, b=b, s=s, c=c, d=d, e=e_par, Phi=Phi
        )
        return Phi, cache

    # ------------- projected model -------------------------------------
    def proj_model(beta):
        Phi, C = build_design(beta)
        alpha, U = solve_linear_block(Phi, y)  # (a,f)
        y_hat = Phi @ alpha
        C.update(alpha=alpha, U=U)
        return y_hat, C

    resid = lambda beta: (proj_model(beta)[0] - y) / y.std()

    # ------------- analytic Jacobian -----------------------------------
    def jacobian(beta):
        y_hat, C = proj_model(beta)
        Phi, (a_hat, _), U = C["Phi"], C["alpha"], C["U"]
        H, H2, phi, phi_p, b = (C["H"], C["H2"], C["phi"], C["phi_p"], C["b"])

        s, c, d, e_par = beta
        dH_ds = db_ds_c(s, inner) * phi
        dH_dc = b * phi_p * x_std
        dH_dd = b * phi_p
        dH_de = np.ones_like(H)

        # ∂/∂β [a·H²] = 2·a·H·∂H/∂β
        col = lambda dH: 2 * a_hat * H * dH
        orth = lambda v: v - U @ (U.T @ v)

        J = np.empty((N, 4))
        J[:, 0] = orth(col(dH_ds))
        J[:, 1] = orth(col(dH_dc))
        J[:, 2] = orth(col(dH_dd))
        J[:, 3] = orth(col(dH_de))
        return J / y.std()

    # ------------- outer optimiser -------------------------------------
    if method.lower() == "lbfgs":
        objective = lambda b: np.mean((proj_model(b)[0] - y) ** 2) / np.var(y)
        x0 = rng.uniform(-0.5, 0.5, 4)
        bounds = [(-7, 7), (-7, 7), (-5, 5), (-5, 5)]
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 600},
        )
    elif method.lower() == "nlls":
        beta0 = rng.uniform(-0.5, 0.5, 4)
        res = least_squares(
            resid, beta0, jac=jacobian, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15
        )
    else:
        raise ValueError("method must be 'lbfgs' or 'nlls'")

    # ------------- assemble parameter vectors --------------------------
    y_hat, C_fin = proj_model(res.x)
    a_std, f_std = C_fin["alpha"]
    s_hat, c_hat, d_hat, e_hat = res.x
    b_hat = b_from_s_c(s_hat, inner)
    theta_std = np.array([a_std, b_hat, c_hat, d_hat, e_hat, f_std])

    # ------------- error metrics ---------------------------------------
    rel_mse_train = np.mean((y_hat - y) ** 2) / np.var(y)

    theta_raw = to_raw_coords(theta_std, x_mu, x_sig)
    theta_can = canonicalise_cubic(theta_raw, inner)  # |c|-rescaling works here

    return dict(
        seed=seed,
        method=method.lower(),
        success=res.success,
        message=res.message if hasattr(res, "message") else res.status,
        rel_mse_train=rel_mse_train,
        params_std=theta_std,
        params_raw=theta_raw,
        params_can=theta_can,
        result=res,
        y_hat_train=y_hat,
    )


FIT_ROUTINE = {
    "cube": fit_once_cube,
    "sin": fit_once_sin,
    "log": fit_once_log,
    "recip": fit_once_recip,
    "exp": fit_once_exp,
    "square": fit_once_square,
}


def fit_stacked(fun_name, x, y, n_restarts=N_RESTARTS, solver_dict=None, fit_dict=None):
    """
    Fit a model five times with fresh random initialisations and
    return the best one (lowest rel_MSE_train).

    Parameters
    ----------
    fun_name   : tuple(str,str)
        (outer, inner)  – for example ('cube','log').
    x, y       : ndarray
        Data to be fitted.
    n_restarts : int
        Number of random initialisations.
    solver_dict: dict or None
        Maps (outer, inner) → 'lbfgs' | 'nlls'.
        If None, defaults to a global SOLVER_CHOICE.
    fit_dict   : dict or None
        Maps  outer → fit_once_function.
        If None, defaults to global FIT_ROUTINE.

    Returns
    -------
    tuple (params_best, rel_mse_train_best)
        params_best are returned exactly as the winning fit_once_* emits
        them (usually 'params_std' or 'params_can' – adapt if needed).
    """
    outer, inner = fun_name

    # --- look up solver and worker ------------------------------------
    if solver_dict is None:
        solver_dict = SOLVER_CHOICE  # global default
    solver = solver_dict[(outer, inner)]

    if fit_dict is None:
        fit_dict = FIT_ROUTINE  # global default
    fit_fun = fit_dict[outer]

    # --- stats for x ---------------------------------------------------
    x_mu, x_sig = x.mean(), x.std()

    # --- multi-start optimisation -------------------------------------
    best_train = np.inf
    best_params = np.ones(6)

    for seed in range(n_restarts):
        try:
            res = fit_fun(
                seed=seed, x=x, y=y, x_mu=x_mu, x_sig=x_sig, inner=inner, method=solver
            )
            if res["rel_mse_train"] < best_train:
                best_train = res["rel_mse_train"]
                best_params = res["params_raw"]

        except Exception as e:
            # So far encountered np.LinalgError and type error
            traceback.print_exc()

    """if fun_name == ("exp", "sin"):
        plt.scatter(x, y, label="true")
        a,b,c,d,e,f=best_params
        pred = lambda x: a*np.exp(b*np.sin(c*x+d)+e)+f
        plt.scatter(x, pred(x), label="pred",s=1)
        plt.legend()
        plt.show()
        print("params in plot:", best_params)"""
    return best_params, best_train


# begin non-stacked fitting block
def fit_linear_map(seed, x, y, true_params=None):
    """
    Closed-form OLS fit of y ≈ a*x + b  to 1-D data.

    Parameters
    ----------
    seed : int
        Kept only to satisfy the requested interface; it is *not* used
        because the closed-form solution is deterministic.
    x, y : 1-D numpy arrays
        Training data.
    true_params : tuple or None
        (a0, b0) – ground-truth parameters.  If supplied, an additional
        relative MSE on an out-of-domain grid is returned; otherwise it is
        reported as np.nan.

    Returns
    -------
    dict
        {
          'params_raw'   : np.array([a_hat, b_hat]),
          'rel_mse_train': relative MSE on the training data,
          'rel_mse_ext'  : relative MSE on the extrapolation grid
                           (np.nan if true_params is None)
        }
    """

    # ---------------------------------------------------------------
    # 1. Closed-form least-squares solution
    # ---------------------------------------------------------------
    # Design matrix:  X = [x, 1]
    X = np.vstack([x, np.ones_like(x)]).T  # shape (n, 2)

    # Normal equations solution: theta = (X^T X)^{-1} X^T y
    theta = np.linalg.lstsq(X, y, rcond=None)[0]  # [a_hat, b_hat]
    a_hat, b_hat = theta

    # ---------------------------------------------------------------
    # 2. Training error
    # ---------------------------------------------------------------
    y_hat = a_hat * x + b_hat
    rel_mse_train = np.mean((y_hat - y) ** 2)  # / np.var(y)

    # ---------------------------------------------------------------
    # 3. Optional out-of-domain (extrapolation) error
    # ---------------------------------------------------------------
    if true_params is not None:
        a0, b0 = true_params
        x_ext = np.linspace(x.min() - 2.0, x.max() + 2.0, 100)

        y_true_ext = a0 * x_ext + b0
        y_fit_ext = a_hat * x_ext + b_hat

        rel_mse_ext = np.mean((y_fit_ext - y_true_ext) ** 2)  # / np.var(y_true_ext)
    else:
        rel_mse_ext = np.nan

    # ---------------------------------------------------------------
    # 4. Return result dictionary
    # ---------------------------------------------------------------
    return theta, rel_mse_train  # , rel_mse_ext


# ---------- sine ----------------------------------------------------------
def sin_val(z):
    return np.sin(z)


def sin_der(z):
    return np.cos(z)


# ---------- logarithm -----------------------------------------------------
def log_val(z, eps=1e-12):
    zc = np.clip(np.abs(z), eps, None)  # avoid log(0)
    return np.log(zc)


def log_der(z, eps=1e-12):
    zc = np.clip(np.abs(z), eps, None)
    return 1.0 / zc


# ---------- exponential ---------------------------------------------------
def exp_val(z, lo=-50.0, hi=50.0):
    return np.exp(np.clip(z, lo, hi))


def exp_der(z, lo=-50.0, hi=50.0):
    return np.exp(np.clip(z, lo, hi))  # derivative equals value


# ---------- reciprocal ----------------------------------------------------
def recip_val(z, clip_val=1e-2, hi=1e5):
    zc = np.clip(z, -hi, hi)
    small = np.abs(zc) < clip_val
    zc[small] = clip_val * np.sign(zc[small])
    return 1.0 / zc


def recip_der(z, clip_val=1e-2, hi=1e5):
    zc = np.clip(z, -hi, hi)
    small = np.abs(zc) < clip_val
    zc[small] = clip_val * np.sign(zc[small])
    return -1.0 / zc**2


# ---------- square --------------------------------------------------------
def square_val(z, hi=1e4):
    zc = np.clip(z, -hi, hi)
    return zc**2


def square_der(z, hi=1e4):
    zc = np.clip(z, -hi, hi)
    return 2.0 * zc


# ---------- cube ----------------------------------------------------------
def cube_val(z, hi=1e3):
    zc = np.clip(z, -hi, hi)
    return zc**3


def cube_der(z, hi=1e3):
    zc = np.clip(z, -hi, hi)
    return 3.0 * zc**2


# -------------------------------------------------------------------------
# unified dictionary  name → (value-function , derivative-function)
# -------------------------------------------------------------------------
VAL_MAP = {
    "sin": (sin_val, sin_der),
    "log": (log_val, log_der),
    "exp": (exp_val, exp_der),
    "recip": (recip_val, recip_der),
    "square": (square_val, square_der),
    "cube": (cube_val, cube_der),
}


# ---------------------------------------------------------------------
#  generic single-model fitter (two non-linear variables)
# ---------------------------------------------------------------------
def _fit_solo_generic(seed, x, y, f_name, method="lbfgs", true_params=None):
    """
    ONE optimisation run for the 4-parameter model
          y = a · f( b·x + c ) + d
    (variable projection eliminates a and d)

    Returns
    -------
    dict with
        'params_raw'   – 4-vector [a,b,c,d]   (raw-x domain)
        'rel_mse_train'
        'rel_mse_ext'  – if true_params is supplied
    """
    rng = np.random.default_rng(seed)

    f_val, f_der = VAL_MAP[f_name]  # value & derivative
    if f_name == "exp":
        f_name += "_solo"
    '''elif f_name=="recip":
        f_name+="_solo"'''
    N = x.size
    x_mu, x_sig = x.mean(), x.std()
    x_std = (x - x_mu) / x_sig  # standardised abscissa

    # ------------------------------------------------------------------
    # design matrix builder
    # ------------------------------------------------------------------
    def build_design(beta):
        s, c_shift = beta  # β = (signed_log_b , c_shift_std)
        b_val = b_from_s_c(s, f_name)
        z = b_val * x_std + c_shift
        f_z, fp = f_val(z), f_der(z)
        Phi = np.column_stack((f_z, np.ones_like(f_z)))
        cache = dict(f=f_z, fp=fp, b=b_val, z=z, Phi=Phi)
        return Phi, cache

    def proj_model(beta):
        Phi, C = build_design(beta)
        alpha, U = solve_linear_block(Phi, y)  # α = (a, d)
        y_hat = Phi @ alpha
        C.update(alpha=alpha, U=U)
        return y_hat, C

    resid = lambda beta: (proj_model(beta)[0] - y) / y.std()

    # ------------------------------------------------------------------
    # Jacobian wrt β
    # ------------------------------------------------------------------
    def jacobian(beta):
        y_hat, C = proj_model(beta)
        Phi, (a_hat, _), U = C["Phi"], C["alpha"], C["U"]
        fp, b_val = C["fp"], C["b"]
        s, c_shift = beta

        df_ds = fp * db_ds_c(s, f_name) * x_std
        df_dc = fp
        orth = lambda v: v - U @ (U.T @ v)

        J = np.empty((N, 2))
        J[:, 0] = orth(a_hat * df_ds) / y.std()
        J[:, 1] = orth(a_hat * df_dc) / y.std()
        return J

    """def objective_recip(beta):
        y_hat, cache = proj_model(beta)
        mse  = np.mean((y_hat - y)**2)
        H    = cache['z']             # this is b*x_std + c_shift
        penalty = 1e3 * np.mean(np.exp(-np.abs(H)/0.05))  # large when |H|≈0
        return mse + penalty"""

    # ------------------------------------------------------------------
    # outer optimiser
    # ------------------------------------------------------------------
    if method.lower() == "lbfgs":
        obj = lambda b: np.mean((proj_model(b)[0] - y) ** 2) / np.var(y)
        x0 = rng.uniform(-0.5, 0.5, 2)
        bounds = [(-5, 5), (-5, 5)]
        res = minimize(
            obj,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 600},
        )
    else:  # 'nlls'
        beta0 = rng.uniform(-0.5, 0.5, 2)
        res = least_squares(
            resid, beta0, jac=jacobian, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15
        )

    # ------------------------------------------------------------------
    # parameter conversion  (std → RAW-x)
    # ------------------------------------------------------------------
    y_hat, C_fin = proj_model(res.x)
    a_std, d_std = C_fin["alpha"]  # already raw
    s_hat, c_std = res.x
    b_std = b_from_s_c(s_hat, f_name)

    b_raw = b_std / x_sig
    c_raw = c_std - b_std * x_mu / x_sig
    theta_raw = np.array([a_std, b_raw, c_raw, d_std])

    # ------------------------------------------------------------------
    # errors
    # ------------------------------------------------------------------
    rel_mse_train = np.mean((y_hat - y) ** 2) / np.var(y)

    if true_params is not None:
        a0, b0, c0, d0 = true_params
        x_ext = np.linspace(x.min() - 2, x.max() + 2, 100)
        y_true_ext = a0 * f_val(b0 * x_ext + c0) + d0
        y_fit_ext = (
            theta_raw[0] * f_val(theta_raw[1] * x_ext + theta_raw[2]) + theta_raw[3]
        )
        rel_mse_ext = np.mean((y_fit_ext - y_true_ext) ** 2) / np.var(y_true_ext)
    else:
        rel_mse_ext = np.nan  # not requested

    return dict(
        params_raw=theta_raw, rel_mse_train=rel_mse_train, rel_mse_ext=rel_mse_ext
    )


# value-only versions of the six admissible outer maps
def _sin(z):
    return np.sin(z)


def _log(z):
    return np.log(np.clip(np.abs(z), 1e-12, None))


def _exp(z):
    return np.exp(np.clip(z, -50, 50))


def _recip(z):
    zc = np.clip(z, -1e5, 1e5)
    return 1 / zc


def _square(z):
    return z**2


def _cube(z):
    return z**3


_1LAYER_FUNS = {
    "sin": _sin,
    "log": _log,
    "exp": _exp,
    "recip": _recip,
    "square": _square,
    "cube": _cube,
    "x": lambda x: x,
}


def generate_synthetic_data_single(
    fun_name: str = "sin",
    true_params: Sequence[float] | None = None,
    n_points: int = 400,
    x_min: float = -2.0,
    x_max: float = 2.0,
):
    """
    Return a dict analogous to the old generator but for
        y = a·f(b·x + c) + d
    """
    if fun_name not in _1LAYER_FUNS:
        raise KeyError(f"unknown outer function '{fun_name}'")

    if true_params is None:
        true_params = np.random.normal(size=4)  # a,b,c,d

    if not fun_name == "x":
        a0, b0, c0, d0 = true_params

        x = np.linspace(x_min, x_max, n_points)
        f_val = _1LAYER_FUNS[fun_name](b0 * x + c0)
        y = a0 * f_val + d0

        x_mu, x_sig = x.mean(), x.std()
        x_std = (x - x_mu) / x_sig

        return dict(
            x=x,
            y=y,
            x_mu=x_mu,
            x_sig=x_sig,
            x_std=x_std,
            params=true_params,
            fun_name=fun_name,
            n=n_points,
        )
    else:
        a0, b0 = true_params[:2]

        x = np.linspace(x_min, x_max, n_points)
        y = _1LAYER_FUNS[fun_name](a0 * x + b0)

        x_mu, x_sig = x.mean(), x.std()
        x_std = (x - x_mu) / x_sig

        return dict(
            x=x,
            y=y,
            x_mu=x_mu,
            x_sig=x_sig,
            x_std=x_std,
            params=true_params,
            fun_name=fun_name,
            n=n_points,
        )


# ---------------------------------------------------------------------
#  concrete wrappers  (signature identical, no “inner” argument)
# ---------------------------------------------------------------------
def fit_solo_sin(*args, **kw):
    return _fit_solo_generic(*args, f_name="sin", **kw)


def fit_solo_log(*args, **kw):
    return _fit_solo_generic(*args, f_name="log", **kw)


def fit_solo_exp(*args, **kw):
    return _fit_solo_generic(*args, f_name="exp", **kw)


def fit_solo_recip(*args, **kw):
    return _fit_solo_generic(*args, f_name="recip", **kw)


def fit_solo_square(*args, **kw):
    return _fit_solo_generic(*args, f_name="square", **kw)


def fit_solo_cube(*args, **kw):
    return _fit_solo_generic(*args, f_name="cube", **kw)


SOLO_ROUTINE = {
    "sin": fit_solo_sin,
    "log": fit_solo_log,
    "exp": fit_solo_exp,
    "recip": fit_solo_recip,
    "square": fit_solo_square,
    "cube": fit_solo_cube,
    "1/x": fit_solo_recip,
    "x^2": fit_solo_square,
    "x^3": fit_solo_cube,
}


# ---------------------------------------------------------------------
#  public front-end – multi-restart selection
# ---------------------------------------------------------------------
def fit_single(fun_name, x, y, n_restarts=5, method="lbfgs", true_params=None):
    """
    Fit  y = a·f(b·x + c) + d   with 5 random seeds and return the best.

    Parameters
    ----------
    fun_name : str                 'sin'|'log'|'exp'|'recip'|'square'|'cube'
    x, y     : ndarray             data
    n_restarts : int               number of random initialisations
    method   : 'lbfgs' | 'nlls'    outer optimiser

    Returns
    -------
    (params_best, rel_mse_best)
        params_best are returned in RAW-x coordinates (canonicalised).
    """
    if fun_name == "x":
        try:
            return fit_linear_map(42, x, y, true_params)
        except np.linalg.LinAlgError:
            # try one more time
            try:
                return fit_linear_map(42 + 1, x, y, true_params)
            except np.linalg.LinAlgError:
                return np.array([1, 1, 1, 0, 0, 0]), np.inf

    best_mse = np.inf
    best_pars = np.ones(6)
    fit_fun = SOLO_ROUTINE[fun_name]

    for seed in range(n_restarts):
        try:
            res = fit_fun(seed, x, y, method=method, true_params=true_params)
            if res["rel_mse_train"] < best_mse:
                best_mse = res["rel_mse_train"]
                best_pars = res["params_raw"]
                rel_mse_ext = res["rel_mse_ext"]

        except Exception as e:
            traceback.print_exc()
            pass  # choose canonical raw form
    return (
        best_pars,
        best_mse,
    )  # , rel_mse_ext use rel_mse_ext for evaluating fit strategies

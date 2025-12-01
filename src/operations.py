# autora.theorist.darts operations.py with the following additions:
# - safe functions
# - power functions
# - sympy export

import typing
from collections import namedtuple
from fickdich import FuckingContextClass

import sympy
import torch
import torch.nn as nn
from sympy.core.expr import Expr

Genotype = namedtuple("Genotype", "normal normal_concat")
y_th = torch.exp(torch.tensor(6, dtype=torch.float64))#torch.exp(torch.tensor(10, dtype=torch.float64))#**3 possibly adabt wrt max(exp(input), output) plus some air for breathing

def isiterable(p_object: typing.Any) -> bool:
    """
    Checks if an object is iterable.

    Arguments:
        p_object: object to be checked
    """
    try:
        iter(p_object)
    except TypeError:
        return False
    return True


def get_operation_label(
    op_name: str,
    params_org: typing.List,
    decimals: int = 4,
    input_var: str = "x",
    output_format: typing.Literal["latex", "console"] = "console",
) -> str:
    r"""
    Returns a complete string describing a DARTS operation.

    Arguments:
        op_name: name of the operation
        params_org: original parameters of the operation
        decimals: number of decimals to be used for converting the parameters into string format
        input_var: name of the input variable
        output_format: format of the output string (either "latex" or "console")

    Examples:
        >>> get_operation_label("classifier", [1], decimals=2)
        '1.00 * x'
        >>> import numpy as np
        >>> print(get_operation_label("classifier_concat", np.array([1, 2, 3]),
        ...     decimals=2, output_format="latex"))
        x \circ \left(1.00\right) + \left(2.00\right) + \left(3.00\right)
        >>> get_operation_label("classifier_concat", np.array([1, 2, 3]),
        ...     decimals=2, output_format="console")
        'x .* (1.00) .+ (2.00) .+ (3.00)'
        >>> get_operation_label("linear_exp", [1,2], decimals=2)
        'exp(1.00 * x + 2.00)'
        >>> get_operation_label("none", [])
        ''
        >>> get_operation_label("reciprocal", [1], decimals=0)
        '1 / x'
        >>> get_operation_label("linear_reciprocal", [1, 2], decimals=0)
        '1 / (1 * x + 2)'
        >>> get_operation_label("linear_relu", [1], decimals=0)
        'ReLU(1 * x)'
        >>> print(get_operation_label("linear_relu", [1], decimals=0, output_format="latex"))
        \operatorname{ReLU}\left(1x\right)
        >>> get_operation_label("linear", [1, 2], decimals=0)
        '1 * x + 2'
        >>> get_operation_label("linear", [1, 2], decimals=0, output_format="latex")
        '1 x + 2'
        >>> get_operation_label("linrelu", [1], decimals=0)  # Mistyped operation name
        Traceback (most recent call last):
        ...
        NotImplementedError: operation 'linrelu' is not defined for output_format 'console'
    """
    if output_format != "latex" and output_format != "console":
        raise ValueError("output_format must be either 'latex' or 'console'")

    params = params_org.copy()

    format_string = "{:." + "{:.0f}".format(decimals) + "f}"

    classifier_str = ""
    if op_name == "classifier":
        value = params[0]
        classifier_str = f"{format_string.format(value)} * {input_var}"
        return classifier_str

    if op_name == "classifier_concat":
        if output_format == "latex":
            classifier_str = input_var + " \\circ \\left("
        else:
            classifier_str = input_var + " .* ("
        for param_idx, param in enumerate(params):
            if param_idx > 0:
                if output_format == "latex":
                    classifier_str += " + \\left("
                else:
                    classifier_str += " .+ ("

            if isiterable(param.tolist()):
                param_formatted = list()
                for value in param.tolist():
                    param_formatted.append(format_string.format(value))

                for value_idx, value in enumerate(param_formatted):
                    if value_idx < len(param) - 1:
                        classifier_str += value + " + "
                    else:
                        if output_format == "latex":
                            classifier_str += value + "\\right)"
                        else:
                            classifier_str += value + ")"

            else:
                value = format_string.format(param)

                if output_format == "latex":
                    classifier_str += value + "\\right)"
                else:
                    classifier_str += value + ")"

        return classifier_str

    num_params = len(params)

    c = [str(format_string.format(p)) for p in params_org]
    c.extend(["", "", ""])

    if num_params == 1:  # without bias
        if output_format == "console":
            labels = {
                "none": "",
                "add": f"+ {input_var}",
                "subtract": f"- {input_var}",
                "mult": f"{c[0]} * {input_var}",
                "linear": f"{c[0]} * {input_var}",
                "relu": f"ReLU({input_var})",
                "linear_relu": f"ReLU({c[0]} * {input_var})",
                "logistic": f"logistic({input_var})",
                "linear_logistic": f"logistic({c[0]} * {input_var})",
                "exp": f"exp({input_var})",
                "linear_exp": f"exp({c[0]} * {input_var})",
                "reciprocal": f"1 / {input_var}",
                "linear_reciprocal": f"1 / ({c[0]} * {input_var})",
                "ln": f"ln({input_var})",
                "linear_ln": f"ln({c[0]} * {input_var})",
                "cos": f"cos({input_var})",
                "linear_cos": f"cos({c[0]} * {input_var})",
                "sin": f"sin({input_var})",
                "linear_sin": f"sin({c[0]} * {input_var})",
                "tanh": f"tanh({input_var})",
                "linear_tanh": f"tanh({c[0]} * {input_var})",
                "power_two": f"{input_var}**2",
                "linear_power_two": f"({c[0]} * {input_var})**2",
                "power_three": f"{input_var}**3",
                "linear_power_three": f"({c[0]} * {input_var})**3",
                "classifier": classifier_str,
            }
        elif output_format == "latex":
            labels = {
                "none": "",
                "add": f"+ {input_var}",
                "subtract": f"- {input_var}",
                "mult": f"{c[0]} {input_var}",
                "linear": c[0] + "" + input_var,
                "relu": f"\\operatorname{{ReLU}}\\left({input_var}\\right)",
                "linear_relu": f"\\operatorname{{ReLU}}\\left({c[0]}{input_var}\\right)",
                "logistic": f"\\sigma\\left({input_var}\\right)",
                "linear_logistic": f"\\sigma\\left({c[0]} {input_var} \\right)",
                "exp": f"+ e^{input_var}",
                "linear_exp": f"e^{{{c[0]} {input_var} }}",
                "reciprocal": f"\\frac{{1}}{{{input_var}}}",
                "linear_reciprocal": f"\\frac{{1}}{{{c[0]} {input_var} }}",
                "ln": f"\\ln\\left({input_var}\\right)",
                "linear_ln": f"\\ln\\left({c[0]} {input_var} \\right)",
                "cos": f"\\cos\\left({input_var}\\right)",
                "linear_cos": f"\\cos\\left({c[0]} {input_var} \\right)",
                "sin": f"\\sin\\left({input_var}\\right)",
                "linear_sin": f"\\sin\\left({c[0]} {input_var} \\right)",
                "tanh": f"\\tanh\\left({input_var}\\right)",
                "linear_tanh": f"\\tanh\\left({c[0]} {input_var} \\right)",
                "power_two": f"{input_var}^2",
                "linear_power_two": f"({c[0]} {input_var})^2",
                "power_three": f"{input_var}^3",
                "linear_power_three": f"({c[0]} {input_var})^3",
                "classifier": classifier_str,
            }
    else:  # with bias
        if output_format == "console":
            labels = {
                "none": "",
                "id": f"id({input_var})",
                "add": f"+ {input_var}",
                "subtract": f"- {input_var}",
                "mult": f"{c[0]} * {input_var}",
                "linear": f"{c[0]} * {input_var} + {c[1]}",
                "relu": f"ReLU({input_var})",
                "linear_relu": f"ReLU({c[0]} * {input_var} + {c[1]} )",
                "logistic": f"logistic({input_var})",
                "linear_logistic": f"logistic({c[0]} * {input_var} + {c[1]})",
                "exp": f"exp({input_var})",
                "linear_exp": f"exp({c[0]} * {input_var} + {c[1]})",
                "reciprocal": f"1 / {input_var}",
                "linear_reciprocal": f"1 / ({c[0]} * {input_var} + {c[1]})",
                "ln": f"ln({input_var})",
                "linear_ln": f"ln({c[0]} * {input_var} + {c[1]})",
                "cos": f"cos({input_var})",
                "linear_cos": f"cos({c[0]} * {input_var} + {c[1]})",
                "sin": f"sin({input_var})",
                "linear_sin": f"sin({c[0]} * {input_var} + {c[1]})",
                "tanh": f"tanh({input_var})",
                "linear_tanh": f"tanh({c[0]} * {input_var} + {c[1]})",
                "power_two": f"{input_var}**2",
                "linear_power_two": f"({c[0]} * {input_var} + {c[1]})**2",
                "power_three": f"{input_var}**3",
                "linear_power_three": f"({c[0]} * {input_var} + {c[1]})**3",
                "classifier": classifier_str,
            }
        elif output_format == "latex":
            labels = {
                "none": "",
                "id": f"id({input_var})",
                "add": f"+ {input_var}",
                "subtract": f"- {input_var}",
                "mult": f"{c[0]} * {input_var}",
                "linear": f"{c[0]} {input_var} + {c[1]}",
                "relu": f"\\operatorname{{ReLU}}\\left( {input_var}\\right)",
                "linear_relu": f"\\operatorname{{ReLU}}\\left({c[0]}{input_var} + {c[1]} \\right)",
                "logistic": f"\\sigma\\left( {input_var} \\right)",
                "linear_logistic": f"\\sigma\\left( {c[0]} {input_var} + {c[1]} \\right)",
                "exp": f"e^{input_var}",
                "linear_exp": f"e^{{ {c[0]} {input_var} + {c[1]} }}",
                "reciprocal": f"\\frac{{1}}{{{input_var}}}",
                "linear_reciprocal": f"\\frac{{1}} {{ {c[0]}{input_var} + {c[1]} }}",
                "ln": f"\\ln\\left({input_var}\\right)",
                "linear_ln": f"\\ln\\left({c[0]} {input_var} + {c[1]} \\right)",
                "cos": f"\\cos\\left({input_var}\\right)",
                "linear_cos": f"\\cos\\left({c[0]} {input_var} + {c[1]} \\right)",
                "sin": f"\\sin\\left({input_var}\\right)",
                "linear_sin": f"\\sin\\left({c[0]} {input_var} + {c[1]} \\right)",
                "tanh": f"\\tanh\\left({input_var}\\right)",
                "linear_tanh": f"\\tanh\\left({c[0]} {input_var} + {c[1]} \\right)",
                "power_two": f"{input_var}^2",
                "linear_power_two": f"({c[0]} * {input_var} + {c[1]})^2",
                "power_three": f"{input_var}^3",
                "linear_power_three": f"({c[0]} * {input_var} + {c[1]})^3",
                "classifier": classifier_str,
            }

    if op_name not in labels:
        raise NotImplementedError(
            f"operation '{op_name}' is not defined for output_format '{output_format}'"
        )

    return labels[op_name]


def get_operation_as_sympy(
    op_name: str,
    params_org: typing.List,
    input_var: Expr,
) -> Expr:
    r"""
    Returns a sympy expression describing a DARTS operation.

    Arguments:
        op_name: name of the operation
        params_org: original parameters of the operation
        input_var: sympy expression of the input node
    """
    # convention: c for list of constants
    c = params_org
    num_params = len(c)
    # safe versions tend to take forever in sympy
    if "safe" in op_name:# and (not "reciprocal" in op_name):
        op_name=op_name[5:]
    if "smooth" in op_name:
        op_name=op_name[:-7]
    if "ramped" in op_name:
        op_name=op_name[:-7]

    def logistic(x):
        return 1 / (1 + sympy.exp(-x))

    epsilon = 1e-10
    if num_params == 2:
        labels = {
            "linear": lambda input_var, c: c[0] * input_var + c[1],
            "linear_relu": lambda input_var, c: sympy.Piecewise(
                (c[0] * input_var + c[1], c[0] * input_var + c[1] > 0), (0, True)
            ),
            "linear_logistic": lambda input_var, c: logistic(c[0] * input_var + c[1]),
            "linear_exp": lambda input_var, c: sympy.exp(c[0] * input_var + c[1]),
            "linear_reciprocal": lambda input_var, c: 1 / (c[0] * input_var + c[1]),
            "linear_ln": lambda input_var, c: sympy.log(c[0] * input_var + c[1]+epsilon),
            "linear_cos": lambda input_var, c: sympy.cos(c[0] * input_var + c[1]),
            "linear_sin": lambda input_var, c: sympy.sin(c[0] * input_var + c[1]),
            "linear_tanh": lambda input_var, c: sympy.tanh(c[0] * input_var + c[1]),
            "linear_power_two": lambda input_var, c: (c[0] * input_var + c[1]) ** 2,
            "linear_power_three": lambda input_var, c: (c[0] * input_var + c[1]) ** 3,
        }
    else:
        labels = {
            "id": lambda inpu_var: input_var,
            "none": lambda input_var: sympy.Integer(0),
            "add": lambda input_var: input_var,
            "subtract": lambda input_var: -input_var,
            "mult": lambda input_var, c: c[0] * input_var,
            "linear": lambda input_var, c: c[0] * input_var,
            "relu": lambda input_var: sympy.Piecewise(
                (input_var, input_var > 0), (0, True)
            ),
            "linear_relu": lambda input_var, c: sympy.Piecewise(
                (c[0] * input_var, c[0] * input_var > 0), (0, True)
            ),
            "logistic": lambda input_var, c: logistic(input_var),
            "linear_logistic": lambda input_var, c: logistic(c[0] * input_var),
            "exp": lambda input_var: sympy.exp(input_var),
            "linear_exp": lambda input_var, c: sympy.exp(c[0] * input_var),
            "reciprocal": lambda input_var: 1 / input_var if (not input_var==0) else 0,
            "linear_reciprocal": lambda input_var, c: 1 / (c[0] * input_var),
            #"ln": lambda input_var: sympy.Piecewise(
            #    (sympy.log(input_var+epsilon), input_var > 0), (0, True)
            #),
            "ln": lambda input_var: sympy.log(input_var+epsilon),
            "linear_ln": lambda input_var, c: sympy.log(c[0] * input_var+epsilon),
            "cos": lambda input_var: sympy.cos(input_var),
            "linear_cos": lambda input_var, c: sympy.cos(c[0] * input_var),
            "sin": lambda input_var: sympy.sin(input_var),
            "linear_sin": lambda input_var, c: sympy.sin(c[0] * input_var),
            "tanh": lambda input_var: sympy.tanh(input_var),
            "linear_tanh": lambda input_var, c: sympy.tanh(c[0] * input_var),
            "power_two": lambda input_var: input_var**2,
            "linear_power_two": lambda input_var, c: (c[0] * input_var) ** 2,
            "power_three": lambda input_var: input_var**3,
            "linear_power_three": lambda input_var, c: (c[0] * input_var) ** 3,
            #"safe_reciprocal": lambda input_var: 1 / input_var if (not input_var==0) else y_th,
            "safe_reciprocal": lambda input_var: sympy.Piecewise(
                (y_th, (0 <= input_var) & (input_var < 1/y_th)),
                (-y_th, (0 > input_var) & (input_var > -1/y_th)),
                (1/input_var, True),
            ),
        }

    if op_name not in labels:
        raise NotImplementedError(f"operation '{op_name}' is not defined")
    if len(c):
        function = labels[op_name](input_var, c)
    else:
        function = labels[op_name](input_var)

    return function


class Identity(nn.Module):
    """
    A pytorch module implementing the identity function.

    $$
    x = x
    $$
    """

    def __init__(self):
        """
        Initializes the identify function.
        """
        super(Identity, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the identity function.

        Arguments:
            x: input tensor
        """
        return x


class NegIdentity(nn.Module):
    """
    A pytorch module implementing the inverse of an identity function.

    $$
    x = -x
    $$
    """

    def __init__(self):
        """
        Initializes the inverse of an identity function.
        """
        super(NegIdentity, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the inverse of an identity function.

        Arguments:
            x: input tensor
        """
        return -x


class Exponential(nn.Module):
    """
    A pytorch module implementing the exponential function.

    $$
    x = e^x
    $$
    """

    def __init__(self):
        """
        Initializes the exponential function.
        """
        super(Exponential, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the exponential function.

        Arguments:
            x: input tensor
        """
        return torch.exp(x)

class SafeReciprocal(nn.Module):
    def forward(self, x):
        self.x_th=1/y_th#torch.tensor(1e-10)#utterly arbitrary.tensor([[-2.0, -0.5, 0.0, 2.0, 20.0], [0.2, -20.0, 10.0]], requires_grad=True)
        #inv_x = 1.0 / x
        inv_x_th = 1.0 / self.x_th# i.e. = y_th

        '''result = torch.where(
            x >= 0,
            torch.minimum(inv_x, inv_x_th),
            torch.maximum(inv_x, -inv_x_th)
        )'''
        result=torch.ones_like(x)
        # torch minimum seem to have the same curse as where
        pos_inf=(0 <= x.detach()) & (x.detach() < self.x_th)
        neg_inf=(0 > x.detach()) & (x.detach() > -self.x_th)

        '''if pos_inf.any() or neg_inf.any():
            print("reciprocal underflow/divis by zero safety")
            print(y_th)'''
        result[pos_inf]=y_th
        result[neg_inf]=-y_th
        result[~(pos_inf|neg_inf)]=1/x[~(pos_inf|neg_inf)]
        return result

class SafePower2(nn.Module):
    def forward(self, x):

        x_th=y_th**(1/2)#exp(20)**(1/2)
        #result = torch.where(abs(x) < x_th, x**2, x_th**2)

        result = torch.ones_like(x)
        result[abs(x) < x_th]=x[abs(x) < x_th]**2
        result[~(abs(x) < x_th)]=x_th**2
        return result

class SafePower3(nn.Module):
    def forward(self, x):
        #x_th=785.77#exp(20)**(1/3)
        x_th=y_th**(1/3)
        #result = torch.where(abs(x) < x_th, x**3, torch.tensor(x_th**3, dtype=x.dtype))
        #result = torch.where(abs(x) < x_th, x**3, x_th**3)

        result = torch.ones_like(x)
        result[abs(x) < x_th]=x[abs(x) < x_th]**3
        result[~(abs(x) < x_th)]=x_th**3
        return result
            
class SafeExponential(nn.Module):
    """
    A pytorch module implementing a clamped exponential function.

    $$
    x = e^x
    $$
    """

    def __init__(self, x_th=None):
        """
        Initializes the exponential function.
        """
        super(SafeExponential, self).__init__()
        if x_th is None:
            self.x_th = torch.log(y_th)
        else:
            self.x_th=torch.tensor(x_th)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the exponential function.

        Arguments:
            x: input tensor
        """
        #exp_max_safe = torch.finfo(x.dtype).max  # Max value for the data type
        #exp_limit = torch.log(torch.tensor(exp_max_safe))  # Safe exponentiation limit
        x = torch.clamp(x, max=self.x_th, min=-self.x_th)
        return torch.exp(x)
    
class SignedLog1pParam(nn.Module):
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return a.sign() * torch.log1p(a.abs())
    
class SafeLinearExponential(nn.Module):
    def __init__(self):
        super().__init__()

        linear = nn.Linear(1, 1, bias=True)

        # Insert the re-parametrisation *only* on this layer
        torch.nn.utils.parametrize.register_parametrization(
            linear, "weight", SignedLog1pParam()
        )

        self.op = nn.Sequential(
            linear,
            SafeExponential()          # your existing “clip + exp” module
        )

    def forward(self, x):
        return self.op(x)
    
    
class SafeExponentialSmooth(nn.Module):
    """
    A pytorch module implementing a clamped exponential function.

    $$
    x = e^x
    $$
    """

    def __init__(self, x_th=None):
        """
        Initializes the exponential function.
        """
        super(SafeExponentialSmooth, self).__init__()
        if x_th is None:
            self.x_th = torch.log(y_th)
        else:
            self.x_th=torch.tensor(x_th)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the exponential function.

        Arguments:
            x: input tensor
        """
        #exp_max_safe = torch.finfo(x.dtype).max  # Max value for the data type
        #exp_limit = torch.log(torch.tensor(exp_max_safe))  # Safe exponentiation limit
        #x = torch.exp(x) if x <= self.x_th else torch.exp(self.x_th)*(x-self.x_th)+torch.exp(self.x_th)
        '''if FuckingContextClass.gumbel_softmax_enabled: # for some reason smooth and ramped are allergic to gs
            x = torch.clamp(x, max=self.x_th)
            x = torch.exp(x)
        else:
            x = torch.where(
                x <= self.x_th,
                torch.exp(x),
                torch.exp(self.x_th) * (x - self.x_th) + torch.exp(self.x_th)
            )'''
            
        
                
        x_out=torch.ones_like(x)
        x_out[x<=self.x_th]=torch.exp(x[x<=self.x_th])
        x_out[~(x<=self.x_th)]=torch.exp(self.x_th) * (x[~(x<=self.x_th)] - self.x_th) + torch.exp(self.x_th)
        return x_out
    
    
class SafeExponentialRamped(nn.Module):
    """
    A pytorch module implementing a clamped exponential function.

    $$
    x = e^x
    $$
    """

    def __init__(self, x_th=None, ramp=1e-5):
        """
        Initializes the exponential function.
        """
        super(SafeExponentialRamped, self).__init__()
        if x_th is None:
            self.x_th = torch.log(y_th)
        else:
            self.x_th=torch.tensor(x_th)
        self.ramp=torch.tensor(ramp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the exponential function.

        Arguments:
            x: input tensor
        """
        #exp_max_safe = torch.finfo(x.dtype).max  # Max value for the data type
        #exp_limit = torch.log(torch.tensor(exp_max_safe))  # Safe exponentiation limit
        #print(x)
        '''if FuckingContextClass.gumbel_softmax_enabled:
            x = torch.clamp(x, max=self.x_th)
            x = torch.exp(x)
        else:
            x = torch.where(
                x <= self.x_th,
                torch.exp(x),
                self.ramp * (x - self.x_th) + torch.exp(self.x_th)
            )'''
        x_out=torch.ones_like(x)
        x_out[x<=self.x_th]=torch.exp(x[x<=self.x_th])
        x_out[~(x<=self.x_th)]=self.ramp * (x[~(x<=self.x_th)] - self.x_th) + torch.exp(self.x_th)
        return x_out

class Cosine(nn.Module):
    r"""
    A pytorch module implementing the cosine function.

    $$
    x = \cos(x)
    $$
    """

    def __init__(self):
        """
        Initializes the cosine function.
        """
        super(Cosine, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the cosine function.

        Arguments:
            x: input tensor
        """
        return torch.cos(x)


class Sine(nn.Module):
    r"""
    A pytorch module implementing the sine function.

    $$
    x = \sin(x)
    $$
    """

    def __init__(self):
        """
        Initializes the sine function.
        """
        super(Sine, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the sine function.

        Arguments:
            x: input tensor
        """
        return torch.sin(x)


class Tangens_Hyperbolicus(nn.Module):
    r"""
    A pytorch module implementing the tangens hyperbolicus function.

    $$
    x = \tanh(x)
    $$
    """

    def __init__(self):
        """
        Initializes the tangens hyperbolicus function.
        """
        super(Tangens_Hyperbolicus, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the tangens hyperbolicus function.

        Arguments:
            x: input tensor
        """
        return torch.tanh(x)


class NatLogarithm(nn.Module):
    r"""
    A pytorch module implementing the natural logarithm function.

    $$
    x = \ln(x)
    $$

    """

    def __init__(self):
        """
        Initializes the natural logarithm function.
        """
        super(NatLogarithm, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the natural logarithm function.

        Arguments:
            x: input tensor
        """
        # should be 1/epsilon=y_th
        epsilon = torch.tensor(1e-10)
        x_safe = x.clamp(min=epsilon)
        result = torch.log(x_safe)
        
        """# make sure x is in domain of natural logarithm
        mask = x.clone()
        mask=(x <= epsilon).detach()
        
        '''if mask.any():
            print("ln zero safety")'''
        result = torch.log(x)
        result[mask]=torch.log(epsilon)"""
        #result = torch.log(nn.functional.relu(x) + epsilon) * mask
        
        '''mask = x.clone()
        mask[(x <= 0.0).detach()] = 0
        mask[(x > 0.0).detach()] = 1
        
        if (x<=0.0).any():
            print("ln zero safety")
        epsilon = 1e-10
        result = torch.log(nn.functional.relu(x) + epsilon) * mask'''

        return result


class MultInverse(nn.Module):
    r"""
    A pytorch module implementing the multiplicative inverse.

    $$
    x = \frac{1}{x}
    $$
    """

    def __init__(self):
        """
        Initializes the multiplicative inverse.
        """
        super(MultInverse, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multiplicative inverse.

        Arguments:
            x: input tensor
        """
        return torch.pow(x, -1)


class Zero(nn.Module):
    """
    A pytorch module implementing the zero operation (i.e., a null operation). A zero operation
    presumes that there is no relationship between the input and output.

    $$
    x = 0
    $$
    """

    def __init__(self, stride):
        """
        Initializes the zero operation.
        """
        super(Zero, self).__init__()
        assert stride==1#not implemented
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the zero operation.

        Arguments:
            x: input tensor
        """
        #print("zero safety:", not FuckingContextClass.discrete_enabled)
        if self.stride == 1:
            # gradient of softmax(w)_i*0 tends to produce nans in backward
            if not FuckingContextClass.discrete_enabled:#gumbel_softmax_enabled:
                #print("reached safety")
                return x.mul(1e-12)#0.0)
            else:
                return torch.zeros_like(x)
        raise NotImplementedError
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)
    
class ID(nn.Module):
    """
    A pytorch module implementing the identity operation.

    $$
    x = 0
    $$
    """

    def __init__(self):
        """
        Initializes the zero operation.
        """
        super(ID, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the identity operation.

        Arguments:
            x: input tensor
        """
        return x


class Softplus(nn.Module):
    r"""
    A pytorch module implementing the softplus function:

    $$
    \operatorname{Softplus}(x) = \frac{1}{β} \operatorname{log} \left( 1 + e^{β x} \right)
    $$
    """

    # This docstring is a raw-string (it starts `r"""` rather than `"""`)
    # so backslashes need not be escaped

    def __init__(self):
        """
        Initializes the softplus function.
        """
        super(Softplus, self).__init__()
        # self.beta = nn.Linear(1, 1, bias=False)
        self.beta = nn.Parameter(torch.ones(1))
        # elf.softplus = nn.Softplus(beta=self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the softplus function.

        Arguments:
            x: input tensor
        """
        y = torch.log(1 + torch.exp(self.beta * x)) / self.beta
        # y = self.softplus(x)
        return y


class Softminus(nn.Module):
    """
    A pytorch module implementing the softminus function:

    $$
    \\operatorname{Softminus}(x) = x - \\operatorname{log} \\left( 1 + e^{β x} \\right)
    $$
    """

    # This docstring is a normal string, so backslashes need to be escaped

    def __init__(self):
        """
        Initializes the softminus function.
        """
        super(Softminus, self).__init__()
        # self.beta = nn.Linear(1, 1, bias=False)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the softminus function.

        Arguments:
            x: input tensor
        """
        y = x - torch.log(1 + torch.exp(self.beta * x)) / self.beta
        return y


# defines all the operations. affine is turned off for cuda (optimization prposes)


def operation_factory(name):
    if name == "none":
        return Zero(1)
    elif name == "id":
        return ID()
    elif name == "add":
        return nn.Sequential(Identity())
    elif name == "subtract":
        return nn.Sequential(NegIdentity())
    elif name == "mult":
        return nn.Sequential(
            nn.Linear(1, 1, bias=False),
        )
    elif name == "linear":
        return nn.Sequential(nn.Linear(1, 1, bias=True))
    elif name == "relu":
        return nn.Sequential(
            nn.ReLU(inplace=False),
        )
    elif name == "linear_relu":
        return nn.Sequential(
            nn.Linear(1, 1, bias=True),
            nn.ReLU(inplace=False),
        )
    elif name == "logistic":
        return nn.Sequential(
            nn.Sigmoid(),
        )
    elif name == "linear_logistic":
        return nn.Sequential(
            nn.Linear(1, 1, bias=True),
            nn.Sigmoid(),
        )
    elif name == "exp":
        return nn.Sequential(
            Exponential(),
        )
    elif name == "linear_exp":
        return nn.Sequential(
            nn.Linear(1, 1, bias=True),
            Exponential(),
        )
    ##elif name == "safe_linear_exp":
    #    return nn.Sequential(
    #        nn.Linear(1, 1, bias=True),
    #        SafeExponential(),
    #    )
    elif name == "safe_linear_exp":
        return SafeLinearExponential()
    elif name == "safe_exp":
        return nn.Sequential(
            SafeExponential(),
        )
    elif name == "safe_exp_smooth":
        return nn.Sequential(
            SafeExponentialSmooth(),
        )
    elif name == "safe_exp_ramped":
        return nn.Sequential(
            SafeExponentialRamped(),
        )
    elif name == "safe_linear_exp":
        return nn.Sequential(
            nn.Linear(1, 1, bias=True),
            SafeExponential(),
        )
    elif name == "cos":
        return nn.Sequential(
            Cosine(),
        )
    elif name == "linear_cos":
        return nn.Sequential(
            nn.Linear(1, 1, bias=True),
            Cosine(),
        )
    elif name == "sin":
        return nn.Sequential(
            Sine(),
        )
    elif name == "linear_sin":
        return nn.Sequential(
            nn.Linear(1, 1, bias=True),
            Sine(),
        )
    elif name == "tanh":
        return nn.Sequential(
            Tangens_Hyperbolicus(),
        )
    elif name == "linear_tanh":
        return nn.Sequential(
            nn.Linear(1, 1, bias=True),
            Tangens_Hyperbolicus(),
        )
    elif name == "reciprocal":
        return nn.Sequential(
            MultInverse(),
        )
    elif name == "safe_reciprocal":
        return nn.Sequential(
            SafeReciprocal(),
        )
    elif name == "linear_reciprocal":
        return nn.Sequential(
            nn.Linear(1, 1, bias=False),
            MultInverse(),
        )
    elif name == "safe_linear_reciprocal":
        return nn.Sequential(
            nn.Linear(1, 1, bias=False),
            SafeReciprocal(),
        )
    elif name == "ln":
        return nn.Sequential(
            NatLogarithm(),
        )
    elif name == "linear_ln":
        return nn.Sequential(
            nn.Linear(1, 1, bias=False),
            NatLogarithm(),
        )
    elif name == "softplus":
        return nn.Sequential(
            Softplus(),
        )
    elif name == "linear_softplus":
        return nn.Sequential(
            nn.Linear(1, 1, bias=False),
            Softplus(),
        )
    elif name == "softminus":
        return nn.Sequential(
            Softminus(),
        )
    elif name == "linear_softminus":
        return nn.Sequential(
            nn.Linear(1, 1, bias=False),
            Softminus(),
        )
    elif name == "power_two":

        class Power2(nn.Module):
            def forward(self, x):
                return x**2

        return nn.Sequential(
            Power2(),
        )
    elif name == "safe_power_two_smooth":

        class Power2SafeSmooth(nn.Module):
            def forward(self, x):
                
                x_th=y_th**(1/2)#exp(20)**(1/2)
                # I think this is smart, lets see
                # result = torch.where(abs(x) < x_th, x**2, (x_th + x - x.detach())**2)
                
                result = torch.ones_like(x)
                result[abs(x) < x_th]=x[abs(x) < x_th]**2
                result[~(abs(x) < x_th)]=(x_th + x[~(abs(x) < x_th)] - x[~(abs(x) < x_th)].detach())**2
                
                return result

        return nn.Sequential(
            Power2SafeSmooth(),
        )
        
    elif name == "safe_power_two":
        return nn.Sequential(
            SafePower2(),
        )

    elif name == "linear_power_two":

        class Power2(nn.Module):
            def forward(self, x):
                return x**2

        return nn.Sequential(
            nn.Linear(1, 1, bias=True),
            Power2(),
        )

    elif name == "safe_linear_power_two":

        class Power2(nn.Module):
            def forward(self, x):
                return x**2

        return nn.Sequential(
            nn.Linear(1, 1, bias=True),
            Power2(),
        )
    
    elif name == "power_three":

        class Power3(nn.Module):
            def forward(self, x):
                return x**3

        return nn.Sequential(
            Power3(),
        )
    elif name == "safe_power_three_smooth":

        class Power3SafeSmooth(nn.Module):
            def forward(self, x):
                #x_th=785.77#exp(20)**(1/3)
                x_th=y_th**(1/3)
                #result = torch.where(abs(x) < x_th, x**3, torch.tensor(x_th**3, dtype=x.dtype))
                #result = torch.where(abs(x) < x_th, x**3, (x_th + x - x.detach())**3)
                
                result = torch.ones_like(x)
                result[abs(x) < x_th]=x[abs(x) < x_th]**3
                result[~(abs(x) < x_th)]=(x_th + x[~(abs(x) < x_th)] - x[~(abs(x) < x_th)].detach())**3
                return result

        return nn.Sequential(
            Power3SafeSmooth(),
        )
                
    elif name == "safe_power_three":

        return nn.Sequential(
            SafePower3(),
        )

    elif name == "linear_power_three":

        class Power3(nn.Module):
            def forward(self, x):
                return x**3

        return nn.Sequential(
            nn.Linear(1, 1, bias=True),
            Power3(),
        )
    elif name == "safe_linear_power_three":
        return nn.Sequential(
            nn.Linear(1, 1, bias=True),
            SafePower3(),
        )
    else:
        raise NotImplementedError(f"operation {name=} it not implemented")


# this is the list of primitives actually used,
# and it should be a set of names contained in the OPS dictionary
PRIMITIVES = (
    "none",
    "add",
    "subtract",
    "linear",
    "linear_logistic",
    "mult",
    "linear_relu",
)

# make sure that every primitive is in the OPS dictionary
for name in PRIMITIVES:
    assert operation_factory(name) is not None

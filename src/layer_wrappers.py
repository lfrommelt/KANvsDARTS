from kan import KANLayer, Symbolic_KANLayer
import numpy as np
import torch

from src.fit1d import SYMBOLIC_LIB, SYMBOLIC_TUPLE_LIB, fit_single, fit_stacked

"""class KANLayerPlus(KANLayer):
    #def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5, scale_base_mu=0, scale_base_sigma=1, scale_sp=1, base_fun=..., grid_eps=0.02, grid_range=..., sp_trainable=True, sb_trainable=True, save_plot_data=True, device='cpu', sparse_init=False):
    #    super().__init__(in_dim, out_dim, num, k, noise_scale, scale_base_mu, scale_base_sigma, scale_sp, base_fun, grid_eps, grid_range, sp_trainable, sb_trainable, save_plot_data, device, sparse_init)

    def """


class SymbolicKANLayerPlus(Symbolic_KANLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # overwrite paramst to accomodate for stacked a*f1(b*f2(c*x+d)+e)+f primitives
        device = kwargs.get("device", "cpu")
        self.affine = torch.nn.Parameter(
            torch.zeros(kwargs["out_dim"], kwargs["in_dim"], 6, device=device)
        )
        self.no_grad_neurons = set()

    def zero_grad_neurons(self):
        # Hook into optimizer step
        for idx in self.no_grad_neurons:
            if self.affine.grad is not None:
                self.affine.grad[*idx, :] = 0

    def update_effective_params(self, no_grad_id):
        # Update the list of columns with no gradients
        self.no_grad_neurons.update(no_grad_id)

    def fix_symbolic_explicit(self, i, j, fun_name, params):
        # no-bullshit method for actually fixing (no regression)
        with torch.no_grad():
            if isinstance(fun_name, str):
                self.affine[j][i][:4] = torch.from_numpy(params[:4]).to(
                    dtype=self.affine.dtype, device=self.affine.device
                )
                fun = SYMBOLIC_LIB[fun_name][0]
                # print("fun at skl 216", fun)
                fun_sympy = SYMBOLIC_LIB[fun_name][1]
                fun_avoid_singularity = SYMBOLIC_LIB[fun_name][3]
            elif isinstance(fun_name, tuple):
                self.affine[j][i] = torch.from_numpy(params).to(
                    dtype=self.affine.dtype, device=self.affine.device
                )
                fun = SYMBOLIC_TUPLE_LIB[fun_name][0]
                fun_sympy = SYMBOLIC_TUPLE_LIB[fun_name][1]
                fun_avoid_singularity = SYMBOLIC_TUPLE_LIB[fun_name][3]
            else:
                raise ValueError(
                    f"WTF is {fun_name} supposed to be? type: {type(fun_name)}"
                )

        self.funs_sympy[j][i] = fun_sympy
        self.funs_name[j][i] = fun_name
        self.funs[j][i] = fun
        self.funs_avoid_singularity[j][i] = fun_avoid_singularity

    def fix_symbolic(
        self,
        i,
        j,
        fun_name,
        x=None,
        y=None,
        random=False,
        a_range=(-10, 10),
        b_range=(-10, 10),
        verbose=True,
        grid_number=101,
        n_1d_fits=5,
        **kwargs,
    ):
        """
        fix an activation function to be symbolic

        Args:
        -----
            i : int
                the id of input neuron
            j : int
                the id of output neuron
            fun_name : str
                the name of the symbolic functions
            x : 1D array
                preactivations
            y : 1D array
                postactivations
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of a
            verbose : bool
                print more information if True

        Returns:
        --------
            r2 (coefficient of determination)

        Example 1
        ---------
        >>> # when x & y are not provided. Affine parameters are set to a = 1, b = 0, c = 1, d = 0
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=2)
        >>> sb.fix_symbolic(2,1,'sin')
        >>> print(sb.funs_name)
        >>> print(sb.affine)

        Example 2
        ---------
        >>> # when x & y are provided, fit_params() is called to find the best fit coefficients
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=2)
        >>> batch = 100
        >>> x = torch.linspace(-1,1,steps=batch)
        >>> noises = torch.normal(0,1,(batch,)) * 0.02
        >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
        >>> sb.fix_symbolic(2,1,'sin',x,y)
        >>> print(sb.funs_name)
        >>> print(sb.affine[1,2,:].data)
        """
        if callable(fun_name):
            # if fun_name itself is a function
            fun = fun_name
            fun_sympy = fun_name
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = "anonymous"

            self.funs[j][i] = fun
            self.funs_avoid_singularity[j][i] = fun
            if random == False:
                self.affine.data[j][i] = torch.tensor(
                    [1.0, 0.0, 1.0, 0.0], device=self.device
                )
            else:
                self.affine.data[j][i] = torch.rand(4, device=self.device) * 2 - 1
            return None
        elif isinstance(fun_name, tuple):
            fun = SYMBOLIC_TUPLE_LIB[fun_name][0]
            fun_sympy = SYMBOLIC_TUPLE_LIB[fun_name][1]
            fun_avoid_singularity = SYMBOLIC_TUPLE_LIB[fun_name][3]
            # actualy not r2
            params, r2 = fit_stacked(
                fun_name, x.numpy(), y.numpy(), n_restarts=n_1d_fits
            )
            r2 = torch.tensor(r2)

            # complexity puishment
            r2 *= 10

            with torch.no_grad():
                self.affine[j][i] = torch.from_numpy(params).to(
                    dtype=self.affine.dtype, device=self.affine.device
                )
        elif isinstance(fun_name, str):
            fun = SYMBOLIC_LIB[fun_name][0]
            fun_sympy = SYMBOLIC_LIB[fun_name][1]
            fun_avoid_singularity = SYMBOLIC_LIB[fun_name][3]
            # actualy not r2
            params, r2 = fit_single(
                fun_name, x.numpy(), y.numpy(), n_restarts=n_1d_fits
            )
            r2 = torch.tensor(r2)
            if fun_name == "x":
                params = np.array([1, params[0], params[1], 0])
            with torch.no_grad():
                self.affine[j][i][:4] = torch.from_numpy(params[:4]).to(
                    dtype=self.affine.dtype, device=self.affine.device
                )
        else:
            raise ValueError(
                f'Fuction "{fun_name}" not regcognized ({type(fun_name)} wrong type?)'
            )

        return r2, params

    def forward(self, x, singularity_avoiding=False, y_th=10.0):
        """
        forward

        Args:
        -----
            x : 2D array
                inputs, shape (batch, input dimension)
            singularity_avoiding : bool
                if True, funs_avoid_singularity is used; if False, funs is used.
            y_th : float
                the singularity threshold

        Returns:
        --------
            y : 2D array
                outputs, shape (batch, output dimension)
            postacts : 3D array
                activations after activation functions but before being summed on nodes

        Example
        -------
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, postacts = sb(x)
        >>> y.shape, postacts.shape
        (torch.Size([100, 5]), torch.Size([100, 5, 3]))
        """

        batch = x.shape[0]
        postacts = []

        for i in range(self.in_dim):
            postacts_ = []
            for j in range(self.out_dim):
                if isinstance(self.funs_name[j][i], str):
                    if singularity_avoiding:
                        xij = (
                            self.affine[j, i, 0]
                            * self.funs_avoid_singularity[j][i](
                                self.affine[j, i, 1] * x[:, [i]] + self.affine[j, i, 2],
                                torch.tensor(y_th),
                            )[1]
                            + self.affine[j, i, 3]
                        )
                    else:
                        xij = (
                            self.affine[j, i, 0]
                            * self.funs[j][i](
                                self.affine[j, i, 1] * x[:, [i]] + self.affine[j, i, 2]
                            )
                            + self.affine[j, i, 3]
                        )
                    # print("xij (str)", xij[:3])
                    # print("xij (str)", xij.shape)
                elif isinstance(self.funs_name[j][i], tuple):
                    # original 2*f(0*x+1)+3
                    # new 0*f(1*g(2*x+3)+4)+5
                    if singularity_avoiding:
                        # inner
                        xij = (
                            self.affine[j, i, 1]
                            * self.funs_avoid_singularity[j][i][0](
                                self.affine[j, i, 2] * x[:, [i]] + self.affine[j, i, 3],
                                torch.tensor(y_th),
                            )[1]
                            + self.affine[j, i, 4]
                        )
                        # outer
                        xij = (
                            self.affine[j, i, 0]
                            * self.funs_avoid_singularity[j][i][1](
                                xij, torch.tensor(y_th)
                            )[1]
                            + self.affine[j, i, 5]
                        )
                    else:
                        # inner
                        # print("input:", x[:,[i]][:3])
                        x_ = (
                            self.affine[j, i, 1]
                            * self.funs[j][i][1](
                                self.affine[j, i, 2] * x[:, [i]] + self.affine[j, i, 3]
                            )
                            + self.affine[j, i, 4]
                        )
                        # print("inner (tuple)", x_[:3])
                        # print("inner (tuple)", x_.shape)
                        # outer
                        # print(f"{self.affine[j,i,0]}*{self.funs_name[j][i][1]}({x_[:3]})+{self.affine[j,i,5]}")
                        # print(f"{self.affine[j,i,0]}*{self.funs[j][i][1](x_)[:3]}+{self.affine[j,i,5]}")
                        xij = (
                            self.affine[j, i, 0] * self.funs[j][i][0](x_)
                            + self.affine[j, i, 5]
                        )
                        # print("outer (tuple)", xij[:3])
                        # print("outer (tuple)", xij.shape)
                else:
                    raise ValueError(
                        f"Fun name should ot be {type(self.funs_name[j][i])}\n{self.funs_name[j][i]}"
                    )
                postacts_.append(self.mask[j][i] * xij)
            # print("postacts iternally", *[pa[:3] for pa in postacts_])
            # try:
            postacts.append(torch.stack(postacts_))
            """except Exception as e:
                print(f"{self.mask[j][i]}*{xij}")
                print(*postacts_, sep="\n")
                raise e"""

        postacts = torch.stack(postacts)
        postacts = postacts.permute(2, 1, 0, 3)[:, :, :, 0]
        y = torch.sum(postacts, dim=2)

        return y, postacts

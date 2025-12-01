import importlib
import math
import os
import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sympy
import torch
from src.fit1d import SYMBOLIC_LIB, SYMBOLIC_TUPLE_LIB
from tqdm import tqdm

# import kan

from kan.LBFGS import *

mult_mod = importlib.import_module("kan.MultKAN")


from src.layer_wrappers import SymbolicKANLayerPlus

# Monkey-Patch layer wrappers to the namespace that will be used by kan.MultKAN
mult_mod.Symbolic_KANLayer = SymbolicKANLayerPlus
OriginalMultKAN = mult_mod.MultKAN


class KAN_SR(OriginalMultKAN):
    """
    Wrapper around pykan's KAN implementation, that aims for specifically using
    KANs for symbolic regression and that extends upon the originally proposed
    algortihm in the ways which are presented in the thesis.
    The class is still pretty redundant with respect to its parent.
    """

    def __init__(self, width, real_affine_trainable=False, **kwargs):
        mult_mod.MultKAN = OriginalMultKAN
        super().__init__(width=width, **kwargs)
        mult_mod.MultKAN = KAN_SR
        self.real_affine_trainable = real_affine_trainable
        if "seed" in kwargs:
            self.set_seed(kwargs["seed"])
            self.shuffling_rng = np.random.default_rng(kwargs["seed"])
        else:
            seed = type(self).seed
            self.shuffling_rng = np.random.default_rng(seed)

    @classmethod
    def set_seed(cls, value):
        cls.seed = value

    def fit(
        self,
        dataset,
        opt="LBFGS",
        steps=100,
        log=1,
        lamb=0.0,
        lamb_l1=1.0,
        lamb_entropy=2.0,
        lamb_coef=0.0,
        lamb_coefdiff=0.0,
        update_grid=True,
        grid_update_num=10,
        loss_fn=None,
        lr=1.0,
        start_grid_update_step=-1,
        stop_grid_update_step=50,
        batch=-1,
        metrics=None,
        save_fig=False,
        in_vars=None,
        out_vars=None,
        beta=3,
        save_fig_freq=1,
        img_folder="./video",
        singularity_avoiding=False,
        y_th=1000.0,
        reg_metric="edge_forward_spline_n",
        display_metrics=None,
        verbose=False,
        print_gradients=False,
    ):
        """
        training

        Args:
        -----
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            start_grid_update_step : int
                no grid updates before this training step
            stop_grid_update_step : int
                no grid updates after this training step
            loss_fn : function
                loss function
            lr : float
                learning rate
            batch : int
                batch size, if -1 then full.
            save_fig_freq : int
                save figure every (save_fig_freq) steps
            singularity_avoiding : bool
                indicate whether to avoid singularity for the symbolic part
            y_th : float
                singularity threshold (anything above the threshold is considered singular and is softened in some ways)
            reg_metric : str
                regularization metric. Choose from {'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'}
            metrics : a list of metrics (as functions)
                the metrics to be computed in training
            display_metrics : a list of functions
                the metric to be displayed in tqdm progress bar

        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization
                other metrics specified in metrics

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.plot()
        # Most examples in toturals involve the fit() method. Please check them for useness.
        """

        if opt == "LBFGS" and (
            batch >= 0 and not (batch == len(dataset["train_input"]))
        ):
            warnings.warn(
                "When using LFBGS Optimizer you should not do mini-batching, so full batch is automatically executed"
            )

        if lamb > 0.0 and not self.save_act:
            print("setting lamb=0. If you want to set lamb > 0, set self.save_act=True")

        old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

        pbar = tqdm(range(steps), desc="")  # , ncols=100)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        no_affine_params, affine_params = [], []

        for name, p in self.named_parameters():
            if not p.requires_grad:  # ignore frozen weights
                continue
            if "affine" in name:
                affine_params.append(p)
            else:
                no_affine_params.append(p)

        if self.real_affine_trainable > 0:
            """param_groups = [
            {"params": no_affine_params, "lr": lr},
            {"params": affine_params   , "lr": lr*self.normal_affine_trainable}
            ]"""
            param_groups = [
                {"params": affine_params + no_affine_params, "lr": lr},
            ]

        else:
            param_groups = [
                {"params": no_affine_params, "lr": lr},
            ]

        if isinstance(opt, torch.optim.Optimizer):
            optimizer = opt
            opt = opt.__class__.__name__
        elif opt == "Adam":
            optimizer = torch.optim.Adam(param_groups)
        elif opt == "LBFGS":
            optimizer = LBFGS(
                param_groups,
                history_size=10,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-32,
                tolerance_change=1e-32,
                tolerance_ys=1e-32,
            )

        results = {}
        results["train_loss"] = []
        results["test_loss"] = []
        results["reg"] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset["train_input"].shape[0]:
            batch_size = dataset["train_input"].shape[0]
            batch_size_test = dataset["test_input"].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(
                dataset["train_input"][train_id],
                singularity_avoiding=singularity_avoiding,
                y_th=y_th,
                verbose=verbose,
            )
            train_loss = loss_fn(pred, dataset["train_label"][train_id])
            if self.save_act:
                if reg_metric == "edge_backward":
                    self.attribute()
                if reg_metric == "node_backward":
                    self.node_attribute()
                reg_ = self.get_reg(
                    reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                )
            else:
                reg_ = torch.tensor(0.0)
            objective = train_loss + lamb * reg_
            objective.backward()
            if verbose:
                print("Gradients stored in optimizer's parameter groups:")
                for param_group in optimizer.param_groups:
                    for param in param_group["params"]:
                        if param.grad is not None:
                            print(f"Parameter: {param.shape}")
                            print(f"Gradient: {param.grad}")

            # Replace NaN gradients via optimizer
            # todo: debug line "value = (x" in spline.py instead
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        if print_gradients:
                            print("shape:", param.shape)
                            print("grad pre nan_to_num:", param.grad)
                        if torch.any(torch.isnan(param.grad)):
                            # Replace NaN values in the gradient with zero
                            param.grad = torch.nan_to_num(param.grad)
            if print_gradients:
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        print("grad:\n", param.grad)

            for symbolic_layer in self.symbolic_fun:
                symbolic_layer.zero_grad_neurons()

            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        for _ in pbar:

            if _ == steps - 1 and old_save_act:
                self.save_act = True

            if save_fig and _ % save_fig_freq == 0:
                save_act = self.save_act
                self.save_act = True

            # train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            train_id = np.arange(len(dataset["train_input"]))

            self.shuffling_rng.shuffle(train_id)
            train_ids = [
                train_id[i * batch_size : (i + 1) * batch_size]
                for i in range(int(np.ceil(len(dataset["train_input"]) / batch_size)))
            ]
            test_id = np.random.choice(
                dataset["test_input"].shape[0], batch_size_test, replace=False
            )

            if (
                _ % grid_update_freq == 0
                and _ < stop_grid_update_step
                and update_grid
                and _ >= start_grid_update_step
            ):
                self.update_grid(dataset["train_input"])

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                for batch_id in train_ids:
                    pred = self.forward(
                        dataset["train_input"][batch_id],
                        singularity_avoiding=singularity_avoiding,
                        y_th=y_th,
                        verbose=verbose,
                    )
                    train_loss = loss_fn(pred, dataset["train_label"][batch_id])
                    if self.save_act:
                        if reg_metric == "edge_backward":
                            self.attribute()
                        if reg_metric == "node_backward":
                            self.node_attribute()
                        reg_ = self.get_reg(
                            reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                        )
                    else:
                        reg_ = torch.tensor(0.0)
                    loss = train_loss + lamb * reg_
                    optimizer.zero_grad()
                    loss.backward()

                    if verbose:
                        print("Gradients stored in optimizer's parameter groups:")
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                if param.grad is not None:
                                    print(f"Parameter: {param.shape}")
                                    print(f"Gradient: {param.grad}")

                    # Replace NaN gradients via optimizer
                    # todo: debug line "value = (x" in spline.py instead
                    for group in optimizer.param_groups:
                        for param in group["params"]:
                            if param.grad is not None:
                                if print_gradients:
                                    print("shape:", param.shape)
                                    print("grad pre nan_to_num:", param.grad)
                                if torch.any(torch.isnan(param.grad)):
                                    # Replace NaN values in the gradient with zero
                                    param.grad = torch.nan_to_num(param.grad)
                    if print_gradients:
                        for group in optimizer.param_groups:
                            for param in group["params"]:
                                print("grad:\n", param.grad)

                    for symbolic_layer in self.symbolic_fun:
                        symbolic_layer.zero_grad_neurons()

                    optimizer.step()

            test_loss = loss_fn_eval(
                self.forward(dataset["test_input"][test_id]),
                dataset["test_label"][test_id],
            )

            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results["train_loss"].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results["test_loss"].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results["reg"].append(reg_.cpu().detach().numpy())

            if _ % log == 0:
                if display_metrics == None:
                    pbar.set_description(
                        "| train: %.2e | test: %.2e | reg: %.2e | "
                        % (
                            torch.sqrt(train_loss).cpu().detach().numpy(),
                            torch.sqrt(test_loss).cpu().detach().numpy(),
                            reg_.cpu().detach().numpy(),
                        )
                    )
                else:
                    string = ""
                    data = ()
                    for metric in display_metrics:
                        string += f" {metric}: %.2e |"
                        try:
                            results[metric]
                        except:
                            raise Exception(f"{metric} not recognized")
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)

            if save_fig and _ % save_fig_freq == 0:
                self.plot(
                    folder=img_folder,
                    in_vars=in_vars,
                    out_vars=out_vars,
                    title="Step {}".format(_),
                    beta=beta,
                )
                plt.savefig(
                    img_folder + "/" + str(_) + ".jpg", bbox_inches="tight", dpi=200
                )
                plt.close()
                self.save_act = save_act

        self.log_history("fit")
        # revert back to original state
        self.symbolic_enabled = old_symbolic_enabled
        return results

    def forward(self, x, singularity_avoiding=False, y_th=10.0, verbose=False):
        """
        forward pass

        Args:
        -----
            x : 2D torch.tensor
                inputs
            singularity_avoiding : bool
                whether to avoid singularity for the symbolic branch
            y_th : float
                the threshold for singularity

        Returns:
        --------
            None

        Example1
        --------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> x = torch.rand(100,2)
        >>> model(x).shape

        Example2
        --------
        >>> from kan import *
        >>> model = KAN(width=[1,1], grid=5, k=3, seed=0)
        >>> x = torch.tensor([[1],[-0.01]])
        >>> model.fix_symbolic(0,0,0,'log',fit_params_bool=False)
        >>> print(model(x))
        >>> print(model(x, singularity_avoiding=True))
        >>> print(model(x, singularity_avoiding=True, y_th=1.))
        """
        x = x[:, self.input_id.long()]
        assert x.shape[1] == self.width_in[0]

        # cache data
        self.cache_data = x

        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.acts_premult = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_spline = []
        self.subnode_actscale = []
        self.edge_actscale = []
        # self.neurons_scale = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        if verbose:
            print(f"inputs:\n{x[:5]}")

        for l in range(self.depth):

            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](
                x,
            )
            # print(preacts, postacts_numerical, postspline)

            # if self.symbolic_enabled == True:
            x_symbolic, postacts_symbolic = self.symbolic_fun[l](
                x, singularity_avoiding=singularity_avoiding, y_th=y_th
            )
            # else:
            #    x_symbolic = torch.zeros_like(x)#0.
            #    postacts_symbolic = torch.zeros_like(x)#0.

            if verbose:
                print(f"layer {l}")
                print(f"postsplines:\n{postspline.shape}\n{postspline[:10]}")
                print(
                    f"residuals:\n{postacts_numerical.shape}\n{postacts_numerical[:10]}"
                )
                print(f"symbolic:\n{postacts_symbolic}")

            x = x_numerical + x_symbolic
            # print("output components")
            # print("postact_numerical", postacts_numerical[:3])
            # print("postact_symbolic", postacts_symbolic[:3])
            # print("x_numerical", x_numerical[:3])
            # print("x_symbolic", x_symbolic[:3])
            # print("x_num+x_symb", x[:3])

            if self.save_act:
                # save subnode_scale
                self.subnode_actscale.append(torch.std(x, dim=0).detach())

            # subnode affine transform
            # print("affine transforms", self.subnode_scale[l][None,:], self.subnode_bias[l][None,:], sep="\n")
            x = self.subnode_scale[l][None, :] * x + self.subnode_bias[l][None, :]
            # print("output_transformed", x[:3])

            if self.save_act:
                postacts = postacts_numerical + postacts_symbolic

                # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
                # grid_reshape = self.act_fun[l].grid.reshape(self.width_out[l + 1], self.width_in[l], -1)
                input_range = torch.std(preacts, dim=0) + 0.1
                output_range_spline = torch.std(
                    postacts_numerical, dim=0
                )  # for training, only penalize the spline part
                output_range = torch.std(
                    postacts, dim=0
                )  # for visualization, include the contribution from both spline + symbolic
                # save edge_scale
                self.edge_actscale.append(output_range)

                self.acts_scale.append((output_range / input_range).detach())
                self.acts_scale_spline.append(output_range_spline / input_range)
                self.spline_preacts.append(preacts.detach())
                self.spline_postacts.append(postacts.detach())
                self.spline_postsplines.append(postspline.detach())

                self.acts_premult.append(x.detach())

            # multiplication
            dim_sum = self.width[l + 1][0]
            dim_mult = self.width[l + 1][1]

            if self.mult_homo == True:
                for i in range(self.mult_arity - 1):
                    if i == 0:
                        x_mult = (
                            x[:, dim_sum :: self.mult_arity]
                            * x[:, dim_sum + 1 :: self.mult_arity]
                        )
                    else:
                        x_mult = x_mult * x[:, dim_sum + i + 1 :: self.mult_arity]

            else:
                for j in range(dim_mult):
                    acml_id = dim_sum + np.sum(self.mult_arity[l + 1][:j])
                    for i in range(self.mult_arity[l + 1][j] - 1):
                        if i == 0:
                            x_mult_j = x[:, [acml_id]] * x[:, [acml_id + 1]]
                        else:
                            x_mult_j = x_mult_j * x[:, [acml_id + i + 1]]

                    if j == 0:
                        x_mult = x_mult_j
                    else:
                        x_mult = torch.cat([x_mult, x_mult_j], dim=1)

            if self.width[l + 1][1] > 0:
                x = torch.cat([x[:, :dim_sum], x_mult], dim=1)

            # x = x + self.biases[l].weight
            # node affine transform
            x = self.node_scale[l][None, :] * x + self.node_bias[l][None, :]
            # print("after mult", x[:3])
            self.acts.append(x.detach())

        return x

    # this is the function fitting, awkward naming
    def fix_symbolic(
        self,
        l,
        i,
        j,
        fun_name,
        fit_params_bool=True,
        a_range=(-10, 10),
        b_range=(-10, 10),
        verbose=True,
        random=False,
        log_history=True,
        fix_params=False,
        x=None,
        y=None,
        n_1d_fits=5,
        **kwargs,
    ):
        """
        set (l,i,j) activation to be symbolic (specified by fun_name)

        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
            fun_name : str
                function name
            fit_params_bool : bool
                obtaining affine parameters through fitting (True) or setting default values (False)
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of b
            verbose : bool
                If True, more information is printed.
            random : bool
                initialize affine parameteres randomly or as [1,0,1,0]
            log_history : bool
                indicate whether to log history when the function is called

        Returns:
        --------
            None or r2 (coefficient of determination)

        Example 1
        ---------
        >>> # when fit_params_bool = False
        >>> model = KAN(width=[2,5,1], grid=5, k=3)
        >>> model.fix_symbolic(0,1,3,'sin',fit_params_bool=False)
        >>> print(model.act_fun[0].mask.reshape(2,5))
        >>> print(model.symbolic_fun[0].mask.reshape(2,5))

        Example 2
        ---------
        >>> # when fit_params_bool = True
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=1.)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model(x) # obtain activations (otherwise model does not have attributes acts)
        >>> model.fix_symbolic(0,1,3,'sin',fit_params_bool=True)
        >>> print(model.act_fun[0].mask.reshape(2,5))
        >>> print(model.symbolic_fun[0].mask.reshape(2,5))
        """
        if not fit_params_bool:
            self.symbolic_fun[l].fix_symbolic(
                i, j, fun_name, verbose=verbose, random=random, **kwargs
            )
            r2 = None
        else:
            if None in (x, y):
                x = self.acts[l][:, i]
                y = self.spline_postacts[l][:, j, i]
            mask = self.act_fun[l].mask
            # y = self.postacts[l][:, j, i]
            r2, params = self.symbolic_fun[l].fix_symbolic(
                i,
                j,
                fun_name,
                x,
                y,
                a_range=a_range,
                b_range=b_range,
                verbose=verbose,
                n_1d_fits=n_1d_fits,
            )
            if mask[i, j] == 0:
                # should not reach
                r2 = -1e8
                sglkllksf
        self.set_mode(l, i, j, mode="s")

        if log_history:
            self.log_history("fix_symbolic")

        if fix_params:
            self.symbolic_fun[l].update_effective_params([(j, i)])
        return r2, params

    @torch.no_grad
    def prune_minimally(
        self, trainset, v=False, local=True, per_neuron=True, semi_minimal=False
    ):
        self.forward(trainset)
        self.attribute()
        if v:
            print(self.edge_scores)

        if not local:
            minimal_edge = np.inf
            for layer in range(len(self.edge_scores)):
                # killing layers not possible, yet
                if self.act_fun[layer].mask.data.bool().sum().item() > 1:
                    minimal_edge = min(
                        (
                            edge
                            for edge in self.edge_scores[layer][
                                self.act_fun[layer].mask.data.bool().permute(1, 0)
                            ]
                        )
                    )

            if math.isinf(minimal_edge):
                print(f"cannot prune further")
                return None

            print(f"pruning score {minimal_edge}")
            self = self.prune_edge(threshold=minimal_edge, log_history=True)

        else:
            ratio = 0.0
            index = None
            for layer in range(0, self.depth):

                layer_mask = self.act_fun[layer].mask.data.T
                symbolic_mask = self.symbolic_fun[layer].mask

                if not per_neuron:
                    masked_layer = torch.sort(
                        self.edge_scores[layer][
                            ((layer_mask - symbolic_mask) > 0).bool()
                        ]
                    )[0]
                    if v:
                        print("layer_mask", layer_mask)
                        print("symbolic_mask", symbolic_mask)
                        print("masked_layer", masked_layer)
                    if len(masked_layer) == 1:
                        continue
                    new_ratio = min(masked_layer[1:] / masked_layer[:-1])
                    if new_ratio > ratio:
                        index = (
                            layer,
                            *(
                                (self.edge_scores[layer] == masked_layer[0]).nonzero()[
                                    0
                                ]
                            ),
                        )
                else:
                    masked_layer = self.edge_scores[layer][
                        ((layer_mask - symbolic_mask) > 0).bool()
                    ]
                    if v:
                        print()
                        print("layer_mask", layer_mask)
                        print("symbolic_mask", symbolic_mask)
                        print("masked_layer", masked_layer)
                    for i, neuron in enumerate(self.edge_scores[layer].T):

                        masked_neuron = torch.sort(
                            neuron[
                                ((layer_mask[:, i] - symbolic_mask[:, i]) > 0).bool()
                            ]
                        )[0]

                        if v:
                            print("masked_neuron", masked_neuron)
                        if len(masked_neuron) == 0:
                            continue
                        elif len(masked_neuron) == 1:
                            # ratio is now second smallest overall edge divided by actual neuron's one
                            if len(masked_layer) >= 2:
                                new_ratio = (
                                    torch.sort(masked_layer)[0][1] / masked_neuron[0]
                                )
                            else:
                                continue  # never prune last connection
                            highest_index = 0
                        else:
                            new_ratio, highest_index = torch.max(
                                masked_neuron[1:] / masked_neuron[:-1], dim=0
                            )

                        if v:
                            print("new_ratio", new_ratio)
                        if new_ratio > ratio:
                            if not semi_minimal:
                                index = (
                                    layer,
                                    i,
                                    (neuron == masked_neuron[0]).nonzero()[0][0],
                                )
                            else:
                                index = [
                                    (layer, i, neuron_index)
                                    for neuron_index in (
                                        neuron <= masked_neuron[highest_index]
                                    ).nonzero()[:, 0]
                                ]
                            ratio = new_ratio

            if not index is None:
                if v:
                    print(f"pruning: {index}")
                    try:
                        print("prune layer mask:", self.act_fun[index[0]].mask)
                    except TypeError:
                        print("prune layer mask:", self.act_fun[index[0][0]].mask)

                if not semi_minimal:
                    self.act_fun[index[0]].mask[index[1], index[2]] = 0.0
                else:
                    for i in index:
                        self.act_fun[i[0]].mask[i[1], i[2]] = 0.0

                if v:
                    print(
                        "new mask:",
                        *[self.act_fun[l].mask for l in range(self.depth)],
                        sep="\n",
                    )

        # prune nodes that became obsolete
        self.forward(trainset)
        self.attribute()
        if v:
            print(self.edge_scores)
        self = self.prune_node(threshold=1e-8)

        # prune edges that became obsolete after node pruning
        self.attribute()
        self = self.prune_edge(threshold=1e-8)
        self.log_history("prune")

        if v:
            print(
                "final mask:",
                *[self.act_fun[l].mask for l in range(self.depth)],
                sep="\n",
            )

        return self

    def prune_edge(self, *args, **kwargs):
        # small wrapper to make the apis in the kan library more consistent
        super().prune_edge(*args, **kwargs)
        return self

    def prune_node(self, *args, **kw):
        print("call prune node (subclasses method)")
        # gotta do this whenever the constructor is implicitly called....
        mult_mod.MultKAN = OriginalMultKAN
        pruned = OriginalMultKAN.prune_node(self, *args, **kw)
        mult_mod.MultKAN = KAN_SR
        # convert the result into the subclass
        pruned.__class__ = KAN_SR
        return pruned

    def minimal_auto_symbolic(
        self, verbose=False, r2_threshold=1e-5, criterion="rmse", n_1d_fits=2, lib=["0"]
    ):
        """
        automatic symbolic regression for all edges

        Args:
        -----
            a_range : tuple
                search range of a
            b_range : tuple
                search range of b
            lib : list of str
                library of candidate symbolic functions
            verbose : int
                larger verbosity => more verbosity
            weight_simple : float
                a weight that prioritizies simplicity (low complexity) over performance (high r2) - set to 0.0 to ignore complexity
            r2_threshold : float
                If r2 is below this threshold, the edge will not be fixed with any symbolic function - set to 0.0 to ignore this threshold
        Returns:
        --------
            None

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,1,1], grid=5, k=3, noise_scale=0.0, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]])+x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.auto_symbolic()
        """
        # list(best_name, best_fun, best_r2, best_c, edge_index)
        results = []

        for l in range(len(self.width_in) - 1):
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l + 1]):
                    if (
                        self.symbolic_fun[l].mask[j, i] > 0.0
                        and self.act_fun[l].mask[i][j] == 0.0
                    ):
                        print(f"skipping ({l},{i},{j}) since already symbolic")
                    elif (
                        self.symbolic_fun[l].mask[j, i] == 0.0
                        and self.act_fun[l].mask[i][j] == 0.0
                    ):
                        self.symbolic_fun[l].fix_symbolic_explicit(
                            i, j, "0", np.array([1, 1, 0, 0])
                        )
                        self.set_mode(l, i, j, mode="s")
                        # self.fix_symbolic_explicit(l, i, j, '0', verbose=verbose > 1, log_history=False)
                        print(f"fixing ({l},{i},{j}) with 0")
                    else:
                        # name, fun, r2, c
                        result = (
                            *self.suggest_symbolic(
                                l,
                                i,
                                j,
                                lib=lib,
                                verbose=verbose > 2,
                                n_1d_fits=n_1d_fits,
                            ),
                            (l, i, j),
                        )
                        results.append(result)
        # print(*results)

        if results:
            # sort based on scores
            # print(*results, sep="\n")
            results.sort(key=lambda t: t[2])

            # list(best_name, best_fun, best_r2, best_params, edge_index)
            best_name, best_fun, best_r2, best_params, idx = results[0]
            # print("fun name mk 2825", best_name)
            l, i, j = idx
            self.symbolic_fun[l].fix_symbolic_explicit(i, j, best_name, best_params)
            self.set_mode(l, i, j, mode="s")

            # if verbose >= 1:
            print(
                f"fixing ({l},{i},{j}) with {best_name}, r2={best_r2}, params={best_params}"
            )

            for name, fun, r2, params, idx in results[1:]:
                # print("fun name mk 2834", name)
                l, i, j = idx
                if r2 <= r2_threshold:
                    if verbose >= 1:
                        print(
                            f"fixing ({l},{i},{j}) with {name}, r2={r2}, params={params}"
                        )
                    self.symbolic_fun[l].fix_symbolic_explicit(i, j, name, params)
                    self.set_mode(l, i, j, mode="s")
                else:
                    break

        self.log_history("minimal_auto_symbolic")

    def suggest_symbolic(
        self,
        l,
        i,
        j,
        a_range=(-10, 10),
        b_range=(-10, 10),
        lib=None,
        topk=None,
        verbose=True,
        r2_loss_fun=lambda x: np.log2(1 + 1e-5 - x),
        c_loss_fun=lambda x: x,
        weight_simple=0.8,
        n_1d_fits=5,
        **kwargs,
    ):
        """
        suggest symbolic function

        Args:
        -----
            l : int
                layer index
            i : int
                neuron index in layer l
            j : int
                neuron index in layer j
            a_range : tuple
                search range of a
            b_range : tuple
                search range of b
            lib : list of str
                library of candidate symbolic functions
            topk : int
                the number of top functions displayed
            verbose : bool
                if verbose = True, print more information
            r2_loss_fun : functoon
                function : r2 -> "bits"
            c_loss_fun : fun
                function : c -> 'bits'
            weight_simple : float
                the simplifty weight: the higher, more prefer simplicity over performance


        Returns:
        --------
            best_name (str), best_fun (function), best_r2 (float), best_c (float)

        Note:
        -----
        If result is zero function, automatic fix

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,1,1], grid=5, k=3, noise_scale=0.0, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]])+x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.suggest_symbolic(0,1,0)
        """
        # c_loss_fun=self.echo_and_print
        r2s = []
        cs = []
        if lib == None:
            symbolic_lib = {**SYMBOLIC_LIB, **SYMBOLIC_TUPLE_LIB}
        else:
            symbolic_lib = {}
            for item in lib:
                symbolic_lib[item] = {**SYMBOLIC_LIB, **SYMBOLIC_TUPLE_LIB}[item]
        if topk is None:
            topk = len(symbolic_lib)

        params_list = []
        # getting r2 and complexities
        for name, content in symbolic_lib.items():
            r2, params = self.fix_symbolic(
                l,
                i,
                j,
                name,
                a_range=a_range,
                b_range=b_range,
                verbose=False,
                log_history=False,
                n_1d_fits=n_1d_fits,
                **kwargs,
            )
            if r2 == -1e8:  # zero function (why???)
                r2s.append(-1e8)
            else:
                r2s.append(r2.item())
                self.unfix_symbolic(l, i, j, log_history=False)
            params_list.append(params)

        loss = np.array(r2s)

        sorted_ids = np.argsort(loss)[:topk]
        loss = loss[sorted_ids][:topk]
        params_list = [params_list[i] for i in sorted_ids]

        topk = np.minimum(topk, len(symbolic_lib))

        if verbose == True:
            # print results in a dataframe
            results = {}
            results["function"] = [
                list(symbolic_lib.items())[sorted_ids[i]][0] for i in range(topk)
            ]
            results["fitting r2"] = r2s[:topk]
            results["complexity"] = cs[:topk]
            results["total loss"] = loss[:topk]

            df = pd.DataFrame(results)
            print(df)

        best_name = list(symbolic_lib.items())[sorted_ids[0]][0]
        best_fun = list(symbolic_lib.items())[sorted_ids[0]][1]
        best_r2 = loss[0]
        best_params = params_list[0]

        return best_name, best_fun, best_r2, best_params

    # Overwritten because
    # - missing kwarg "simplify"
    # - added stacked primitives
    def symbolic_formula(
        self, var=None, normalizer=None, output_normalizer=None, simplify=True
    ):
        """
        get symbolic formula

        Args:
        -----
            var : None or a list of sympy expression
                input variables
            normalizer : [mean, std]
            output_normalizer : [mean, std]

        Returns:
        --------
            None

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,1,1], grid=5, k=3, noise_scale=0.0, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]])+x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.auto_symbolic()
        >>> model.symbolic_formula()[0][0]
        """

        symbolic_acts = []
        symbolic_acts_premult = []
        x = []

        def ex_round(ex1, n_digit):
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, n_digit))
            return ex2

        # define variables
        if var == None:
            for ii in range(1, self.width[0][0] + 1):
                exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                exec(f"x.append(x{ii})")
        elif isinstance(var[0], sympy.Expr):
            x = var
        else:
            x = [sympy.symbols(var_) for var_ in var]

        x0 = x

        if normalizer != None:
            mean = normalizer[0]
            std = normalizer[1]
            x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

        symbolic_acts.append(x)

        for l in range(len(self.width_in) - 1):
            num_sum = self.width[l + 1][0]
            num_mult = self.width[l + 1][1]
            y = []
            for j in range(self.width_out[l + 1]):
                yj = 0.0
                for i in range(self.width_in[l]):
                    params = self.symbolic_fun[l].affine[j, i]
                    name = self.symbolic_fun[l].funs_name[j][i]
                    sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]
                    if isinstance(name, str):
                        a, b, c, d, _, _ = params
                        try:
                            yj += a * sympy_fun(b * x[i] + c) + d
                        except Exception as e:
                            print(
                                "make sure all activations need to be converted to symbolic formulas first!"
                            )
                            raise e
                    elif isinstance(name, tuple):
                        a, b, c, d, e, f = params
                        try:
                            # inner
                            inner = b * sympy_fun[1](c * x[i] + d) + e
                            # outer
                            yj += a * sympy_fun[0](inner) + f
                        except Exception as e:
                            print(
                                "make sure all activations need to be converted to symbolic formulas first!"
                            )
                            raise e

                yj = self.subnode_scale[l][j] * yj + self.subnode_bias[l][j]
                if simplify == True:
                    y.append(sympy.simplify(yj))
                else:
                    y.append(yj)

            symbolic_acts_premult.append(y)

            mult = []
            for k in range(num_mult):
                if isinstance(self.mult_arity, int):
                    mult_arity = self.mult_arity
                else:
                    mult_arity = self.mult_arity[l + 1][k]
                for i in range(mult_arity - 1):
                    if i == 0:
                        mult_k = y[num_sum + 2 * k] * y[num_sum + 2 * k + 1]
                    else:
                        mult_k = mult_k * y[num_sum + 2 * k + i + 1]
                mult.append(mult_k)

            y = y[:num_sum] + mult

            for j in range(self.width_in[l + 1]):
                y[j] = self.node_scale[l][j] * y[j] + self.node_bias[l][j]

            x = y
            symbolic_acts.append(x)

        if output_normalizer != None:
            output_layer = symbolic_acts[-1]
            means = output_normalizer[0]
            stds = output_normalizer[1]

            assert len(output_layer) == len(
                means
            ), "output_normalizer does not match the output layer"
            assert len(output_layer) == len(
                stds
            ), "output_normalizer does not match the output layer"

            output_layer = [
                (output_layer[i] * stds[i] + means[i]) for i in range(len(output_layer))
            ]
            symbolic_acts[-1] = output_layer

        self.symbolic_acts = [
            [symbolic_acts[l][i] for i in range(len(symbolic_acts[l]))]
            for l in range(len(symbolic_acts))
        ]
        self.symbolic_acts_premult = [
            [symbolic_acts_premult[l][i] for i in range(len(symbolic_acts_premult[l]))]
            for l in range(len(symbolic_acts_premult))
        ]

        out_dim = len(symbolic_acts[-1])

        if simplify:
            return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
        else:
            return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0

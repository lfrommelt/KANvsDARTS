import json
import time
import numpy as np
import traceback
import threading
import matplotlib.pyplot as plt
from utils import PRIMITIVES, rmse, nrmse, round_and_simplify, sl_rmse, rmspe, r2, make_vectorized, NumericTensorJSONEncoder
from pathlib import Path
import pickle
import sympy as sp

import os
import sys

'''import warnings
warnings.simplefilter("error")'''
#os.environ["PYTHONWARNINGS"] = "error"

# sys.path.append(os.path.normpath(os.getcwd() + "/autora-theorist-darts/src"))


from train_darts import train_new_darts
#print(train)
import utils

success_metric=nrmse
metric_norm="std"
success_threshold=0.2
scale_steps=1.0#/100
# baaaaaad
success_threshold = 0.18#??

success_threshold = 0.002#last good recovery that still included exponential error was slightly above

success_threshold = 0.1#for std-nrmse, slightly above was bad in k1 results, slightly below was minimally off
success_threshold = 0.01

comment = "lower th (0.01)"

# List the variable names we want to record
keys = ('success_metric', 'metric_norm', 'success_threshold',
        'scale_steps', 'comment')

# Grab their current values in *this* scope.  
# If any name is missing, Python will raise a KeyError/NameError as requested.
vals = {k: locals()[k] for k in keys}

# Ensure the target directory exists
Path('Results').mkdir(exist_ok=True)

# Overwrite the file with a tidy, human-readable report
with open('Results/description.txt', 'w', encoding='utf-8') as f:
    for k, v in vals.items():
        f.write(f'{k.replace("_", " ").title():20}: {v}\n')
        
# Import the W&B Python Library and log into W&B

import os

#os.environ["WANDB_DEBUG"] = "0"

#from autora.theorist.darts.model_search import Network

'''
???
def create_int_bins(start, end, n_edges):
    bin_edges = np.linspace(start, end, n_edges)
    # Round edges to integers
    bin_edges = np.round(bin_edges).astype(int)
    return bin_edges
'''

def stop_on_enter():
    input("Press [Enter] at any time to stop the sweep...\n")
    print("Exiting on user request.")
    # os._exit(1) exits the whole process, works even in wandb agent
    os._exit(1)

# random mapping index -> primitive
primitives=np.array(PRIMITIVES)
np.random.seed(0)
np.random.shuffle(primitives)


# darts hps
config2={
  'arch_discretization': 'softmax',
  'arch_learning_rate_max': 0.6473364090755143,
  'arch_momentum': 0.0017282364618274,
  'batch_size': 20,
  'coeff_discretization': 'max',
  'finetune_epochs': 10,
  'param_learning_rate_max': 0.0006232415860704,
  'param_momentum': 1.9581913155e-06,
  'primitives': ["none","power_two","power_three","exp","ln","reciprocal","sin"],
  'ratio_train_val': 1.0,
  'safety': 'ramped',
  'steps': 720,
  'train_output_layer': False,}

config2={'Name': 'fallen-sweep-100',
 'arch_discretization': 'softmax',
 'arch_learning_rate_max': 0.8893474836112625,
 'arch_momentum': 0.0066695070207013,
 'arch_weight_decay': 0.0002014087677275,
 'batch_size': 7,
 'coeff_discretization': 'max',
 'coeff_lr_min_scale': 1.0,#0.0009878495458311,
 'finetune_epochs': 60,#8
 'init_range': 1.0120530236610816,
 'param_learning_rate_max': 5e-4,#3.373269557527869e-09,
 'param_momentum': 1.509044847784784e-09,
 'param_weight_decay': 7.2956488739e-06,
 'pruning': 'none',
 'ratio_train_val': 4.0,#2.0,
 'safety': 'safe',
 'size': 2,
 'train_output_layer': True,
 'primitives': ["none","power_two","power_three","exp","ln","reciprocal","sin","id"],
 'loss_fn':"mse"}

config4={
  'arch_discretization': 'softmax',
  'arch_learning_rate_max': 1.6284305674366029,
  'arch_momentum': 0.0022416242214032,
  'batch_size': 20,
  'coeff_discretization': 'max',
  'finetune_epochs': 10,
  'param_learning_rate_max': 0.0012008188033979,
  'param_momentum': 0.0027466356242729,
  'primitives': ["none","power_two","power_three","exp","ln","reciprocal","sin"],
  'ratio_train_val': 1.0,
  'safety': 'ramped',
  'steps': 720,
  'train_output_layer': False,
}

config6={
  'arch_discretization': 'softmax',
  'arch_learning_rate_max': 9.928562713532394,
  'arch_momentum': 8.784944800579377e-09,
  'batch_size': 20,
  'coeff_discretization': 'max',
  'finetune_epochs': 10,
  'param_learning_rate_max': 0.0654276023875558,
  'param_momentum': 5.318695973457925e-10,
  'primitives': ["none","power_two","power_three","exp","ln","reciprocal","sin"],
  'ratio_train_val': 1.0,
  'safety': 'ramped',
  'steps': 720,
  'train_output_layer': False,}
    
config4=config2
config6=config2

configs=(config2, config4, config6)

sizes=[1,2,3,4,5]
        
# conditions
c_multiplication = (False, True)
c_prior_knowledge = (True, False)
c_linear_transformations = (False, True)

# for simulatin results for debugging
test_rng=np.random.default_rng(42)

def main():
    for multiplication in c_multiplication:
        for prior_knowledge in c_prior_knowledge:
            for linear_transformations in c_linear_transformations:
                if (not linear_transformations) and (not multiplication):
                    continue
                # k=0 should be trivial
                for k in range(0,5):
                    all_failed=True
                    for n_v in range(1,4):                            
                        # no mult between at least two independent variables
                        if multiplication and (n_v==1):
                            continue
                        for seed in range(3):
                            # for now we concider each seed an individual trial (iid problem!)
                            dataset_folder=Path(f"datasets/mult_{multiplication}-lin_{linear_transformations}")
                            with open(dataset_folder/f"equations_k{k}_nv{n_v}.txt", "rt") as f:
                                equations=[line.rstrip() for line in f]
                            for combination in range(6):
                                # for continuing when interrupted
                                '''params={
                                     'k': 1,
                                     'multiplication': False,
                                     'prior_knowledge': True,
                                     'linear_transformations': False,}
                                local_vars=locals()
                                if all(local_vars[key] == val for key, val in params.items()):
                                    continue'''
                                '''if not(linear_transformations) and not(multiplication):
                                    continue'''
                                    
                                # start a trial
                                path=f"Results/mult_{multiplication}-prior_{prior_knowledge}-lin_{linear_transformations}/k_{k}-nv_{n_v}-comb{combination}-seed{seed}.json"
                                print(f"--------------------{path}--------------------")
                                path=Path(path)
                                run_logs=[]

                                # for k=0, all structures are the same
                                if k==0 and (combination>0) and (not linear_transformations):
                                    continue
                                for size in range(max(k-1,1),k+3):
                                    # start a run
                                    if size in (1,2,3):
                                        config=config2.copy()
                                    elif size in (4,5):
                                        config=config4.copy()
                                    else:
                                        config=config6.copy()

                                    ground_truth=equations[combination]
                                    print("gt:", ground_truth)

                                    config["size"]=size
                                    config["dataset"]=f"datasets_k{k}_nv{n_v}/{combination}.pkl"
                                    config["training_seed"]=seed
                                    config["combination"]=combination

                                    if linear_transformations:
                                        config["train_output_layer"]=True
                                    if prior_knowledge:
                                        all_primitives=config["primitives"]
                                        actual_primitives=[]
                                        #actual_primitives=["none"]
                                        if not linear_transformations:
                                            actual_primitives.append("id")
                                            actual_primitives.append("none")
                                        '''else:
                                            actual_primitives.append("linears")'''
                                        if multiplication:
                                            # one way to get a KAR
                                            actual_primitives.append("power_two")

                                        for primitive in all_primitives:
                                            str_=primitive
                                            if str_=="power_two":
                                                str_="**2"
                                            elif str_=="power_three":
                                                str_="**3"
                                            elif str_=="reciprocal":
                                                str_="1/"
                                            if str_ in ground_truth:
                                                actual_primitives.append(primitive)

                                        config["primitives"]=actual_primitives
                                    print(config["primitives"])
                                    if linear_transformations or multiplication:
                                        config["primitives"]=["linear_"+primitive if not((primitive=="none") or (primitive=="id")) else primitive for primitive in config["primitives"]]
                                        config["primitives"].append("linear")
                                    print(config["primitives"])


                                    config["param_learning_rate_min"]= config["param_learning_rate_max"]*config["coeff_lr_min_scale"]

                                    train_ratio=config["ratio_train_val"]
                                    val_ratio=1
                                    test_ratio=(train_ratio+val_ratio)*0.25#one-fifth of data will be test
                                    ratio=(train_ratio, val_ratio, test_ratio)
                                    ratio=np.array(ratio)*(1/min(ratio[:2]))
                                    config["ratio"]=ratio

                                    # load dataset
                                    dataset_path=dataset_folder/config["dataset"]
                                    with open(dataset_path, 'rb') as f:
                                        dataset=pickle.load(f)

                                    print(config["coeff_discretization"])

                                    # output normalize
                                    #print("before normalize")
                                    #print(dataset["train_label"].min())
                                    #print(dataset["train_label"].max())
                                    #x=dataset["train_label"]
                                    #y_mean=x.mean()#(axis=0)
                                    #y_std=x.std()#(axis=0)
                                    #dataset["train_label"]=(x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)

                                    #print("after normalize")
                                    #print(dataset["train_label"].min())
                                    #print(dataset["train_label"].max())

                                    t0 = time.perf_counter()

                                    monitor, model, predicted_equation, outcome = train_new_darts(config, 
                                                                                                  dataset, training_seed=seed, 
                                                                                                  scale_steps=scale_steps, 
                                                                                                  coeff_opti="adam",
                                                                                                 failfast=False,
                                                                                                 debug=False,
                                                                                                 )
                                    print(predicted_equation)
                                    #predicted_equation=predicted_equation*y_std+y_mean
                                    #y=dataset["train_label"]
                                    #dataset["train_label"]=y*y_std+y_mean
                                    # make nans in eval less likely by pruning e.g. 0.0001*log(x_1)
                                    predicted_equation=round_and_simplify(predicted_equation, ndigits=4)
                                    runtime = time.perf_counter() - t0

                                    print(f"runtime: {runtime}")
                                    print("gt:", ground_truth)
                                    #print(type(predicted_equation))
                                    print("pred:",predicted_equation)

                                    symbols = list(sp.symbols([f'x_{i+1}' for i in range(n_v)]))

                                    try:
                                        success_val=success_metric(dataset["extrapolation_label"][:,0], make_vectorized(predicted_equation, symbols)(dataset["extrapolation_input"]), norm=metric_norm)
                                    except Exception as e:
                                        print(type(e), e)
                                        success_val=np.inf

                                        
                                    run_logs.append({
                                        "size":size,
                                        "train_loss":monitor.train_loss,
                                        "val_loss":monitor.val_loss,
                                        "test_loss":monitor.test_loss,
                                        "coefficients":monitor.coefficients.numpy().tolist(),
                                        "alphas":monitor.alphas,
                                        "success_val":success_val,
                                        "prediction": str(predicted_equation),
                                        "runtime": runtime,
                                        "outcome": outcome,
                                    })

                                    '''for (key, val) in run_logs[-1].items():
                                        print(key, type(val))'''
                                    
                                    
                                    '''print(success_val)
                                    print(type(success_val))
                                    print(success_val < success_threshold)
                                    print(monitor.test_loss)
                                    print(run_logs[-1]["test_loss"])'''
                                    
                                    if success_val < success_threshold:
                                        all_failed=False
                                        run_logs[-1]["success"]=True
                                        print("first success!!!")
                                        break

                                    #print("never reach this")
                                    run_logs[-1]["success"]=False

                                trial_log={
                                    "mult":multiplication,
                                    "prior":prior_knowledge,
                                    "lin":linear_transformations,
                                    "k":k,
                                    "nv":n_v,
                                    "comb": combination,
                                    "seed":seed,
                                    "run_logs":run_logs,
                                    "train_loss":run_logs[-1]["train_loss"][-1],
                                    "inter_loss":run_logs[-1]["test_loss"][-1],
                                    "extra_loss":run_logs[-1]["success_val"],
                                    "success":run_logs[-1]["success"],
                                    "success_val": success_val,
                                    "prediction": str(predicted_equation),
                                    "gt":ground_truth,
                                }
                                #print(trial_log["success"])

                                path.parent.mkdir(parents=True, exist_ok=True)
                                with path.open('wt') as f:
                                    json.dump(trial_log, f, cls=NumericTensorJSONEncoder)
                                    
                    if all_failed:
                        break
                    



if __name__ == "__main__":
    main()

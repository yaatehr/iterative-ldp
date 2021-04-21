from typing import List, Any, Tuple
import numpy as np
import os
import matlab.engine
# import math
# my server matlab install is at /usr/local/MATLAB/R2021a
#somehow i have the matlab compiler and sdk and coder on there too
#isntalling with communications toolbox, optimization toolbox, statistics and machine learning toolbox (also the sdks)
#created the symlinks to matlab scripts
#login name Yaateh-ubuntu
# calling matlab from python 

def sigmoid(x: float):
    x = np.copy(x)
    return (1 + np.exp(-x))**-1

def matlab_to_numpy(mlarray):
    return np.array(mlarray._data).reshape(mlarray.size, order='F')



class IDLDP:
    def __init__(self):
        eng = matlab.engine.start_matlab()
        MATLAB_ROOT = eng.matlabroot()
        print(MATLAB_ROOT)

        optimization_path = os.path.join(MATLAB_ROOT, "toolbox", "shared", "optimlib")
        stats_path = os.path.join(MATLAB_ROOT, "toolbox", "shared", "statslib")
        # toolbox_path = os.path.join(MATLAB_ROOT, "toolbox", "shared", "statslib")
        
        eng.addpath(optimization_path)
        eng.addpath(stats_path)
        # print(path.abspath(__file__))
        eng.addpath(os.path.dirname(os.path.abspath(__file__))) #add min opt modules
        # print(eng.fmincon)
        # print(eng.tabulate)
        # print(eng.randsrc)
        self.eng = eng

    @staticmethod
    def create_config_dict(
            privacy_budget: List[float],
            tier_split_percentages: List[float] = None,
            domain_size: int = 100,
            total_records: int = 100000,
        ):
        n_tiers = len(privacy_budget)

        # create default tier split percentage (uniform)
        if not tier_split_percentages:
            tier_split_percentages = np.array([1/n_tiers for i in range(n_tiers)])
            def normalize_list(l):
                remainder = 1 - np.sum(l)
                assert remainder >= 0, "lists to be mnormalized must have magnitude less than or equal to 1"
                l[-1] += remainder
                return l
            tier_split_percentages = normalize_list(tier_split_percentages)
        
        privacy_tier_indices = np.arange(domain_size)
        np.random.shuffle(privacy_tier_indices)
        # print(privacy_tier_indices)
        alpha = (domain_size * np.array(tier_split_percentages)).astype(int)
        alpha[-1] += domain_size - np.sum(alpha)
        split_points = np.cumsum(alpha)[:-1]
        # print(split_points)
        tier_indices = np.split(privacy_tier_indices, split_points)

        return dict(
            privacy_budget= privacy_budget, #W
            n_tiers = n_tiers, #N_lev
            tier_split_percentages= tier_split_percentages, # PROB_
            domain_size=domain_size, #N_LOC
            total_records=total_records, # N_USER
            tier_indices=tier_indices, # W_LIST
            alpha=alpha,
        )

    def min_opt0(self, epsilons: float, **kwargs) -> Tuple[List[float], float]:
        """
        non convex optimization problem for theoretical best perturbatino probabilities. Note the resulting perturbation proabbilites may not sum to 1...? TODO
        Outputs the a and b perturbation probabilities for each tier X and the estimated MSE
        X = [a_1,a_2 .... a_n_tiers, b_1, b_2, ... b_n_tiers]
        returns X, MSE
        """

        args = [privacy_budget, n_tiers, tier_split_percentages, domain_size, total_records, tier_indices, alpha] = list(kwargs.values())
        # MSE = domain_size*np.exp(epsilon/2)/(np.exp(epsilon/2)-1)**2
        output, mse, exitflag = self.eng.min_opt0(matlab.double(epsilons.tolist()), matlab.double(alpha.tolist()), 2, nargout=3) #TODO figure out how to add more returned arguments....
        #TODO if the exitflag does not equal 1, add some safety here...
        x = matlab_to_numpy(output)
        return x, mse    

    def min_opt1(self, epsilons: float, **kwargs) -> Tuple[List[float], float]:
        args = [privacy_budget, n_tiers, tier_split_percentages, domain_size, total_records, tier_indices, alpha] = list(kwargs.values())
        print(epsilons)
        print(alpha)
        output, mse, exitflag = self.eng.min_opt1(matlab.double(epsilons.tolist()), matlab.double(alpha.tolist()), 2, nargout=3) #TODO figure out how to add more returned arguments....
        x = matlab_to_numpy(output)
        print(exitflag)

        return x, mse

    def min_opt2(self, epsilons: float, **kwargs) -> Tuple[List[float], float]:
        args = [privacy_budget, n_tiers, tier_split_percentages, domain_size, total_records, tier_indices, alpha] = list(kwargs.values())
        output, mse, exitflag = self.eng.min_opt2(matlab.double(epsilons.tolist()), matlab.double(alpha.tolist()), 2, nargout=3) #TODO figure out how to add more returned arguments....
        x = matlab_to_numpy(output)
        return x, mse

    def gen_perturbation_probs(
            self,
            epsilon: float,
            privacy_budget: List[float],
            tier_split_percentages: List[float] = None,
            domain_size: int = 100,
            total_records: int = 100000,
            opt_mode: int = 0,
        ):
        assert opt_mode in range(5), "opt mode must be 1,2, or 3"


        config = self.create_config_dict(
            privacy_budget=privacy_budget,
            tier_split_percentages = tier_split_percentages,
            domain_size = domain_size,
            total_records = total_records,
        )

        epsilons = np.array(privacy_budget)*epsilon
        args = [privacy_budget, n_tiers, tier_split_percentages, domain_size, total_records, tier_indices, alpha] = config.values()
        a, b, ind_to_tier = None, None, None
        if opt_mode == 0:
            X, pred_MSE = self.min_opt0(epsilons, **config)
            a = np.ones((n_tiers,1))*sigmoid(epsilon/2)
            b = 1-a
        elif opt_mode == 1:
            X, pred_MSE = self.min_opt1(epsilons, **config)
            a = sigmoid(X)
            b = np.power((1 + np.exp(X)), -1)
            #note this is not symmetric, is this supposed to be? wasn't written like that in their code....
        elif opt_mode == 2:
            X, pred_MSE = self.min_opt2(epsilons, **config)
            a = np.ones((n_tiers, 1))
            b = X
        elif opt_mode == 3:
            # RAPPOR BASIC
            a = .75
            b = .25

        elif opt_mode == 4:
            #OUE BASIC
            a = 1/2 #this is the probabilty a 1 stays a 1. so this is p in the USNIX (they end up being the same....)
            b = 1/(np.exp(epsilon) + 1) # this is flipping 0 to 1, which is equal to q in rapport and USENIX
        else:
            raise Exception("invalid opt mode")

        if opt_mode < 3:
            tier_vals = []
            # print(tier_indices)
            for i in range(len(tier_indices)):
                tier_vals.append([i]*len(tier_indices[i]))

            ind_to_tier = dict(zip(np.concatenate(tier_indices).ravel().tolist(), np.concatenate(tier_vals).ravel().tolist()))
        else:
            ind_to_tier = None

        config["a"] = a
        config["b"] = b
        config["ind_to_tier"] = ind_to_tier

        return X, pred_MSE, config

privacy_budget = [1, 1.2, 2]
tier_split_percentages = [.05, .05, .9]
domain_size = 100
total_records = 10000
opt_mode = 1

IDLDP().gen_perturbation_probs(
  2,
  privacy_budget,
  tier_split_percentages=None,
  domain_size=domain_size,
  total_records = total_records,
  opt_mode = 1,
)

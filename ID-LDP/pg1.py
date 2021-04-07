from typing import List, Any
import numpy as np



# calling matlab from python 
try:
    import matlab.engine
    eng = matlab.engine.start_matlab()
    MATLAB_ROOT = eng.matlabroot()

    optimization_path = os.path.join(MATLAB_ROOT, "toolbox", "shared", "optimlib")
    stats_path = os.path.join(MATLAB_ROOT, "toolbox", "shared", "statslib")
    # toolbox_path = os.path.join(MATLAB_ROOT, "toolbox", "shared", "statslib")
    
    eng.addpath(optimization_path)
    eng.addpath(stats_path)
    print(eng.fmincon)
    print(eng.tabulate)
    print(eng.randsrc)
except Exception as e:
    print(e)
    print("Continuing without matlab")




def create_config_dict(
        privacy_budget: List[float],
        tier_split_percentages: List[float] = None,
        domain_size: int = 100,
        total_records: int = 100000,
    ):
    n_tiers = len(privacy_budget)

    # create default tier split percentage (uniform)
    if not tier_split_percentages:
        tier_split_percentages = [1/n_tiers for i in range(n_tiers)]
        def normalize_list(l):
            remainder = 1 - np.sum(l)
            assert remainder >= 0, "lists to be mnormalized must have magnitude less than or equal to 1"
            l[-1] += remainder
            return l
        tier_split_percentages = normalize_list(tier_split_percentages)
    
    privacy_tier_indices = np.arange(domain_size)
    np.random.shuffle(privacy_tier_indices)
    tier_indices = np.split(privacy_tier_indices, (tier_split_percentages*domain_size)[:-1])
    return dict(
        privacy_budget= privacy_budget, #W
        n_tiers = n_tiers, #N_lev
        tier_split_percentages= tier_split_percentages, # PROB_
        domain_size=domain_size, #N_LOC
        total_records=total_records, # N_USER
        tier_indices=tier_indices, # W_LIST
    )


def sigmoid(x: float):
    x = np.copy(x)
    return (1 + np.exp(-x))**-1



def min_opt0(epsilon: float, **kwargs) -> Tuple[List[float], float]:
    """
    non convex optimization problem for theoretical best perturbatino probabilities. Note the resulting perturbation proabbilites may not sum to 1...? TODO
    Outputs the a and b perturbation probabilities for each tier X and the estimated MSE
    X = [a_1,a_2 .... a_n_tiers, b_1, b_2, ... b_n_tiers]
    returns X, MSE
    """
    gen_constraints = lambda x: nonlcon(x, n_tiers, privacy_budget*epsilon)
    upper_bounds = np.vstack((np.ones(n_tiers, 1), .5*np.ones(n_tiers, 1)))
    lower_bounds = np.vstack((.5*np.ones(n_tiers,1), np.zeros(n_tiers,1)))
    x_0 = np.vstack((.5*np.ones(n_tiers,1), sigmoid(epsilon)))


def nonlcon(x, n_tiers, epsilons):
    pass
    

    
def fun0(x: List[float], **kwargs):
    pass

def fun1(x: List[float], **kwargs):
    pass
def fun2(x: List[float], **kwargs):
    pass

# def fun1(x, **kwargs):
#     pass

# def fun2(x, **kwargs):
#     pass
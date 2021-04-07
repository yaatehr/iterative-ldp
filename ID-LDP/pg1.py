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

    
def fun0(x: List[float], **kwargs):
    alpha = domain_size * np.array(tier_split_percentages)
    A, B = np.split(x, n_tiers)
    quotient = np.divide((B - np.power(B, 2)), (A - np.power(A,2)))
    maxes = np.max(np.divide((1-A)-B, A-B))
    if debug:
        print("debug output, shapes of alpha, quotient, and maxes")
        print(alpha.shape)
        print(quotient.shape)
        print(maxes.shape)
    return alpha@quotient + maxes

def fun1(x: List[float], **kwargs):
    alpha = domain_size * np.array(tier_split_percentages)
    return alpha@np.div(np.exp(x), (np.exp(x)-1)^2)
 
def fun2(x: List[float], **kwargs):
    alpha = domain_size * np.array(tier_split_percentages)
    return alpha@np.div((x - x**2), (.5 - x)**2) + 1


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



def min_opt0(epsilons: float, **kwargs) -> Tuple[List[float], float]:
    """
    non convex optimization problem for theoretical best perturbatino probabilities. Note the resulting perturbation proabbilites may not sum to 1...? TODO
    Outputs the a and b perturbation probabilities for each tier X and the estimated MSE
    X = [a_1,a_2 .... a_n_tiers, b_1, b_2, ... b_n_tiers]
    returns X, MSE
    """
    gen_constraints = lambda x: nonlcon(x, n_tiers, epsilons)
    upper_bounds = np.vstack((np.ones((n_tiers, 1)), .5*np.ones((n_tiers, 1))))
    lower_bounds = np.vstack((.5*np.ones((n_tiers,1)), np.zeros((n_tiers,1))))
    x_0 = np.vstack((.5*np.ones((n_tiers,1)), sigmoid(epsilons)))

    options = eng.optimoptions('fmincon', 'Algorithm', 'sqp')
    [X, FVAL, EXITFLAG] = eng.fmincon(fun0, x_0, [], [], [], [], LB, UB, gen_constraints, options)
    # MSE = domain_size*np.exp(epsilon/2)/(np.exp(epsilon/2)-1)**2
    return X, FVAL


def nonlcon(x, n_tiers, epsilons):
    """
    generate the non linear constraints for opt0
    return c, ceq
    """
    A, B = np.split(x, n_tiers)
    prod = A@(1-B)
    exp = np.exp(np.outer.min(epsilons, epsilons))
    return prod - exp@(B@(1-A)), []
    

def min_opt1(epsilons: float, **kwargs) -> Tuple[List[float], float]:
    A = np.zeros((n_tiers**2, n_tiers))
    B = np.outer.min(epsilons, epsilons).flatten()
    row = lambda x,y: x*n_tiers + y

    for i in range(n_tiers):
        for j in range(n_tiers):
            if i == j:
                A[row(i,j), i] = 2
            else:
                A[row(i,j), i] = 1
                A[row(i,j), j] = 1
    LB = np.zeros((n_tiers,1))
    x_0 = np.min(epsilons)*np.ones((n_tiers,1))/2
    options = eng.optimoptions('fmincon', 'Algorithm', 'sqp')
    [X, FVAL, EXITFLAG] = eng.fmincon(fun0, x_0, A, B, [], [], LB, [], [], options)
    return X, FVAL
 

def min_opt2(epsilons: float, **kwargs) -> Tuple[List[float], float]:
    A = np.zeros((n_tiers**2, n_tiers))
    B = np.ones((n_tiers**2, 1))
    row = lambda x,y: x*n_tiers + y 
    for i in range(n_tiers):
        for j in range(n_tiers):
            if i == j:
                A[row(i,j), i] = 1+np.exp(epsilons[i])
            else:
                A[row(i,j), i] = np.exp(np.min(epsilons[i], epsilons[j]))
                A[row(i,j), j] = 1

    LB = np.zeros(n_tiers, 1)
    UB = .5*np.ones((n_tiers, 1))
    x_0 = np.power(np.exp(np.min(epsilons)))

    options = eng.optimoptions('fmincon', 'Algorithm', 'sqp')
    [X, FVAL, EXITFLAG] = eng.fmincon(fun0, x_0, -1*A, -1*B, [], [], LB, UB, [], options)
    return X, FVAL


def gen_perturbation_probs(
        epsilon: float,
        privacy_budget: List[float],
        tier_split_percentages: List[float] = None,
        domain_size: int = 100,
        total_records: int = 100000,
        opt_mode: int = 0,
    ):
    assert opt_mode in range(3), "opt mode must be 1,2, or 3"
    config = create_config_dict(
        privacy_budget=privacy_budget,
        tier_split_percentages = tier_split_percentages,
        domain_size = domain_size,
        total_records = total_records,
    )
    epsilons = privacy_budget*epsilon
    a, b = None, None
    if opt_mode == 0:
        X, pred_MSE = min_opt0(epsilons, **config)
        a = np.ones((n_tiers,1))*sigmoid(epsilon/2)
        b = 1-a
    elif opt_mode == 1:
        X, pred_MSE = min_opt1(epsilons, **config)
        a = sigmoid(X)
        b = np.power((1 + np.exp(X)), -1)
        #note this is not symmetric, is this supposed to be? wasn't written like that in their code....
    elif opt_mode == 2:
        X, pred_MSE = min_opt1(epsilons, **config)
        a = np.ones((n_tiers, 1))
        b = X
    else:
        raise Exception("invalid opt mode")
    

    return a, b, X, pred_MSE, config



#%matplotlib inline
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import timeit
from functions import *

#The class for Randomized Response:
class Randomized_Response:
    def __init__(self, absz, pri_para, **kwargs): # absz: alphabet size, pri_para: privacy parameter
        self.absz = absz #alphabet size k
        self.exp = math.exp(pri_para) #privacy parameter
        self.flip_prob = (self.absz - 1)/(math.exp(pri_para) + self.absz - 1) #flipping probability to maintain local privacy
    
    def encode_string(self, samples):
        n = len(samples)
        # Start by setting private_samples = samples.
        private_samples_rr = np.copy(samples)
        # Determine which samples need to be noised ("flipped").
        flip = np.random.random_sample(n) < self.flip_prob
        flip_samples = samples[flip]
        # Select new samples uniformly at random to replace the original ones.
        rand_samples = np.random.randint(0, self.absz - 1, len(flip_samples))
        # Shift the samples if needed to avoid sampling the orginal samples.
        rand_samples[rand_samples >= flip_samples] += 1
        # Replace the original samples by the randomly selected ones.
        private_samples_rr[flip] = rand_samples
        return private_samples_rr
    
    def decode_string(self, out_samples, normalization = 0):
        #normalization options: 0: clip and normalize(default)
        #                       1: simplex projection
        #                       else: no nomalization
        n = len(out_samples)
        (counts_rr,temp) = np.histogram(out_samples, range(self.absz+1))
        # Estimate the PMF using the count vector.
        p_rr = (counts_rr / float(n)) * ((self.exp  + self.absz - 1) /(self.exp - 1)) - 1.0 / (self.exp - 1)
        #p_rr = decode_counts(counts_rr, epsilon, n, self.absz)
        # Check if truncation and renormalization is required.

        if normalization == 0: 
            p_rr = probability_normalize(p_rr) #clip and normalize
        if normalization == 1:
            p_rr = project_probability_simplex(p_rr) #simplex projection
        return p_rr
    
class RAPPOR:
    def __init__(self, absz, pri_para, config=None, input_map=None, **kwargs): # absz: alphabet size, pri_para: privacy parameter
        self.absz = absz #alphabet size k
        self.pri_para = pri_para
        self.exp = math.exp(pri_para / 2.0) #privacy parameter
        self.flip_prob = 1/(math.exp(pri_para/2.0) + 1) #flipping probability to maintain local privacy
        self.ind_to_tier = None
        self.a = None
        self.b = None
        # print(config)
        # print(pri_para)
        self.input_map = input_map #if input_map is not None else {i: i for i in range(self.absz)}
        if config is not None:
            # print(config)
            self.update_config(**config)
    # def encode_string(self, samples):
    #     n = len(samples)
    #     users = range(n)
    #     # One-hot encode the input integers.
    #     print(samples)
    #     print(self.absz)
    #     private_samples_rappor = np.zeros((n, self.absz))
    #     private_samples_rappor[users, samples] = 1
    #     # Flip the RAPPOR encoded bits with probability self.flip_prob
    #     flip = np.random.random_sample((n, self.absz)) #fill matrix with random vals [0,1)
    #     return np.logical_xor(private_samples_rappor, flip < self.flip_prob)


    def encode_string(self, samples, config=None):
        if config is not None:
            self.update_config(**config)
        
        n = len(samples)
        if not n:
            return None
        users = range(n)
        # One-hot encode the input integers.
        # print(samples.tolist())
        # print(self.absz)
        inputs = np.vectorize(self.input_map.__getitem__)(samples).astype("uint8") if self.input_map is not None else samples
        # print(inputs)
        private_samples_rappor = np.zeros((n, self.absz),  dtype="uint8")
        private_samples_rappor[users, inputs] = 1
        oue_mode = False
        flip = np.random.random_sample((n, self.absz)) #fill matrix with random vals [0,1)
        if self.a is None or self.b is None:
            self.a = .5 #this is the probabilty a 1 stays a 1. so this is p in the USNIX (they end up being the same....)
            self.b = 1/(np.exp(self.pri_para) + 1)
            oue_mode = True
            # print("A and B are not set, defaulting to OUE")
        if self.input_map is not None or oue_mode:
            # print(b.shape)
            # print(a.shape)
            # print(self.b.shape)
            # print(self.a.shape)
            sample_b_flip = np.tile(self.b, (n,1)).reshape((n,1)).astype("float32")
            # print(sample_b_flip.shape)
            sample_a_1_pr = np.tile(self.a, (n,1)).reshape((n,1)).astype("float32")
        else:
            sample_tiers = np.vectorize(self.ind_to_tier.__getitem__)(samples).astype("uint8")
            # print(self.b.T)
            # print(self.a.T)
            tb =  np.tile(self.b.T.astype("float32"), (n, 1))
            # print(tb.shape)
            ta = np.tile(self.a.T.astype("float32"), (n, 1))
            # print(ta.shape)
            sample_b_flip = tb[users, sample_tiers].reshape((n,1)) #TODO at 100k samples, this tries to allocate 200GiB sized arrays which causes an error
            # print(sample_b_flip.shape)
            # potentially see example here: https://stackoverflow.com/questions/39611045/use-multi-processing-threading-to-break-numpy-array-operation-into-chunks
            sample_a_1_pr = ta[users, sample_tiers].reshape((n,1))

        # print(sample_b_flip.shape)
        # print("begin rappor prints)")
        # print(np.sum(flip))
        np.less(flip, sample_b_flip, out=flip)
        # print(np.sum(flip))
        # print(np.sum(private_samples_rappor))
        np.logical_xor(private_samples_rappor, flip, out=private_samples_rappor)# perturb the b indices
        # print(np.sum(private_samples_rappor))
        private_samples_rappor[np.ix_(users, inputs)] = np.random.random_sample((n,1)) < sample_a_1_pr #perturb the a indices
        # print(np.sum(private_samples_rappor))
        # print("end rappor prints)")
        # print(private_samples_rappor.shape)
        return private_samples_rappor

    def update_config(self, **kwargs):
        args = [privacy_budget, n_tiers, tier_split_percentages, domain_size, total_records, tier_indices, alpha, a, b, ind_to_tier] = list(kwargs.values())
        self.a = a
        self.b = b
        self.ind_to_tier = ind_to_tier


    def encode_string_light(self, samples):
        #return to count vector of rappor responce, which is less memory intensive
        #also return the cumulated time for adding rappor vectors, which should also be considered as decoding time.
        n = len(samples)
        users = range(n)
        time = 0
        counts = np.zeros(self.absz)
        # One-hot encode the input integers.
        for i in range(n):
            private_samples_rappor = np.zeros(self.absz)
            private_samples_rappor[samples[i]] = 1
            # Flip the RAPPOR encoded bits with probability self.flip_prob
            flip = np.random.random_sample(self.absz)
            private_samples_rappor = np.logical_xor(private_samples_rappor, flip < self.flip_prob) 
            start_time = timeit.default_timer() #record adding time
            counts = counts + private_samples_rappor # add rappor responce vector
            time = time + timeit.default_timer() - start_time      
        return counts,time

    def encode_string_compress(self, samples):
        #encode rappor responces into locations of one, which saves communcation budget when eps is large
        n = len(samples)
        out = [0]*n
        # One-hot encode the input integers.
        for i in range(n):
            private_samples_rappor = np.zeros(self.absz)
            private_samples_rappor[samples[i]] = 1
            # Flip the RAPPOR encoded bits with probability self.flip_prob
            flip = np.random.random_sample(self.absz)
            private_samples_rappor = np.logical_xor(private_samples_rappor, flip < self.flip_prob) 
            out[i] = np.where(private_samples_rappor)[0] # get the locations of ones
        out_list = np.concatenate(out)
        return out_list
    
    def decode_counts(self, counts, n, normalization = 0):

        #normalization options: 0: clip and normalize(default)
        #                       1: simplex projection
        #                       else: no nomalization
        # Estimate the PMF using the count vector
        
        p_rappor = (counts / float(n)) * ((self.exp + 1) /(self.exp - 1)) - 1.0 / (self.exp - 1)
        if normalization == 0: 
            p_rappor = probability_normalize(p_rappor) #clip and normalize
        if normalization == 1:
            p_rappor = project_probability_simplex(p_rappor) #simplex projection

        return p_rappor

    def decode_string(self, out_samples, normalization = 0, **kwargs):

        #normalization options: 0: clip and normalize(default)
        #                       1: simplex projection
        #                       else: no nomalization
        # Estimate the PMF using the count vector
        # print(out_samples.shape) 
        if out_samples is None:
            output = np.empty(self.absz)
            output[:] = np.nan
            return output
        n = len(out_samples)
        # print(self.input_map)
        (counts_rr,temp) = np.histogram(out_samples, range(self.absz+1))
        # print(counts_rr)


        #TODO compute the estimate sfor OUE
    # def aggregate(self, priv_data):
    #     """
    #     Used to aggregate privatised data by ue_client.privatise
    #     Args:
    #         priv_data: privatised data from ue_client.privatise
    #     """
    #     self.aggregated_data += priv_data
    #     self.n += 1

    # def _update_estimates(self):
        p_rappor = (counts_rr - n * self.b) / (self.a - self.b)
        # print("p rappor size: \n")
        # print(p_rappor.shape)
        # print(p_rappor)
        # print("\n\n")



        # # This will work for Rappor and optimized rappor but not for OUE
        # p_rappor = (counts / float(n)) * ((self.exp + 1) /(self.exp - 1)) - 1.0 / (self.exp - 1)
        
        if normalization == 0: 
            p_rappor = probability_normalize(p_rappor) #clip and normalize
        if normalization == 1:
            p_rappor = project_probability_simplex(p_rappor) #simplex projection
        # print(p_rappor)
        # raise Exception("over")
        return p_rappor

class MetaEstimator:
    def __init__(self, absz, pri_para, config=None, input_map=None, estimator_class=None, p=10):
        assert estimator_class is not None, 'estimator class must be a valid class'

        self.p = p
        self.epsilon = pri_para / p
        self.estimators = [estimator_class(absz, self.epsilon, config=config, input_map=input_map) for _ in range(p)]

    def encode_string(self, samples, config=None):
        return [e.encode_string(samples, config=config) for e in self.estimators]

    def decode_string(self, encodings, **kwargs):
        distributions = [e.decode_string(encodings[i], **kwargs)for i, e in enumerate(self.estimators)]
        return np.sum(np.vstack(tuple(distributions)), axis=0) / self.p

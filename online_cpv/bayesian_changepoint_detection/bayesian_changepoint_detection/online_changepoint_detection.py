from __future__ import division
import numpy as np
from scipy import stats

def online_changepoint_detection(data, hazard_func, observation_likelihood):
    maxes = np.zeros(len(data) + 1)
    
    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    
    for t, x in enumerate(data):
        t0 = int(len(data)/20)
        if t>t0:
            t=t0
            observation_likelihood.prune(1)
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = observation_likelihood.pdf(x)
        
        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(t+1)))
       
        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)
        
        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t+1] = np.sum( R[0:t+1, t] * predprobs * H)
        
        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])
        
        # Update the parameter sets for each possible run length.
        observation_likelihood.update_theta(x)
    
        maxes[t] = R[:, t].argmax()
    return R, maxes


def constant_hazard(lam, r):
    return 1/lam * np.ones(r.shape)


class StudentT:
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data):
        return stats.t.pdf(x=data, 
                           df=2*self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa+1) / (self.alpha *
                               self.kappa)))

    def update_theta(self, data):
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0, self.beta + (self.kappa * (data -
            self.mu)**2) / (2. * (self.kappa + 1.))))
            
        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0
        
        
class Lomax:
    def __init__(self, alpha, lamda,cutoff=np.inf):
        # not to be confused with the alpha of a Gamma density, or lambda of an exp density
        # I am using notation from Wikipedia https://en.wikipedia.org/wiki/Lomax_distribution
        # where alpha is the shape parameter and lamda is the scale parameter
        self.alpha0 = self.alpha = np.array([alpha])
        self.lamda0 = self.lamda = np.array([lamda])
        self.cutoff=cutoff
    
    def pdf(self, data):
        return stats.lomax.pdf(x=data, loc=0, c=self.alpha, scale=self.lamda)
    
    def update_theta(self, data):
        if data<self.cutoff: 
            alphaT0 = np.concatenate((self.alpha0, self.alpha+1))
        else:
            alphaT0 = np.concatenate((self.alpha0, self.alpha))
        lamdaT0 = np.concatenate((self.lamda0, self.lamda+data))
            
        self.alpha = alphaT0
        self.lamda = lamdaT0
        
    def prune(self, amount):
        self.alpha = self.alpha[:-amount]
        self.lamda = self.lamda[:-amount]
            
     

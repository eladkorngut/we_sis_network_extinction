# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 09:45:26 2018

@author: ohad
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:39:41 2018

@author: metar
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 09:03:17 2018

@author: metar
"""


import numpy as np 
#import matplotlib.pyplot as plt
from scipy.optimize import root
#from scipy import linalg 
from scipy.io import savemat
import sys 
from sympy.utilities.iterables import multiset_permutations
import time
import networkx as nx
import random as rand

def RateEquation(n, N, dn, mu):
    f = np.zeros(n.size)
    for i in range(N.size):
        f[i] =  -(1+D*mu)*n[i] - n[i]**3/((1 - dn[i]**2)*N[i]**2) + (2*n[i]**2)/((1 - dn[i]**2)*N[i]) +mu*sum(n)
    return f
   
def AddingMatrix(d):
    a = np.zeros(d-1)
    a[0] = 1
    b = np.zeros([(d-1),(d-1)])
    i = 0
    for p in multiset_permutations(a):
        b[:, i] = p 
        i = i+1
    add = np.zeros([d, d+d+d*(d-1)])
    add[0:d, 0:d] = np.diag(np.ones(d))
    add[0:d, d:2*d] = np.diag(-np.ones(d))
    for i in range(d):
        add[i, (2*d+(d-1)*i):(2*d+(d-1)*(i+1))] = -np.ones(d-1)
        add[np.arange(d)!=i, (2*d+(d-1)*i):(2*d+(d-1)*(i+1)) ] = b
    
    return add 


def GillespieMC(n, weights , tau, k, N, Alpha, Beta):
    G = nx.random_regular_graph(k,N) # Creates a random graphs with k number of neighbors
    steps_c = np.size(n,1) # Number of simulations
    t = np.zeros(steps_c)
    ng = np.zeros(np.shape(n))
    wg = np.zeros(steps_c)
    it = 10**6
    keep = it 
    j = 0 
    death = 0 

    while 1 : 
        if keep >= it - np.size(n,0):
            r2 = np.random.rand(it)
            r1 = -np.log(np.random.rand(it))
            keep = 0 
        a_vec = Beta * nx.adjacency_matrix(G,nodelist=range(0,N))*n*(np.ones(n.shape)-n)+Alpha*n # Rates matrix each, columns different networks rows different nodes
        a = a_vec.sum(0)
        index = ((a_vec.cumsum(0))/a < r2[keep:keep+steps_c]).sum(0) # First index instance rate for a rate that's above the random number
        t = t + 1/a*r1[keep:keep+steps_c]
        for i in range(index.size): n[index[i], i] = 1 - n[index[i], i] # Change the infect\heal nodes from index array
        keep = keep +  np.size(n, 1)
        # Finds out if a network reached extinction and if so remove it from the dynamical process
        if np.any(n.sum(0)<=0):
            con = n.sum(0)<=0
            y = con.sum()
            death = death + weights[con].sum()
            t = t[~con]
            n = n[~con,:]
            steps_c = np.size(n,1)
            weights = weights[~con]
            wg = wg[:-y]
            ng = ng[:, :-y]
            if steps_c == 0:
               break
        # Finds out if a network has advanced the prerequired time steps (tau) and if stop running its dynamics
        if np.any(t >= tau):
            con =  t >= tau         
            y = con.sum() 
            ng[:,j:j+y] = n[:,con]
            wg[j:j+y] = weights[con]
            t = t[~con]
            n = n[:,~con]
            weights = weights[~con]
            steps_c = np.size(n,1)
            j = j + y 
            if steps_c == 0:
               break
    return ng, wg, death

def resample(n, w, steps):
    Wtot = np.sum(w)
    SortedIdxW = np.argsort(w)
    W = w[SortedIdxW]
    N = n[SortedIdxW, :]
    # split all trajectories that have too much weight
    while np.any(W >= 2*Wtot/steps): 
        IdxL = W >= 2*Wtot/steps
        W = np.hstack([W[~IdxL], W[IdxL]/2, W[IdxL]/2])
        N = np.vstack([N, N[IdxL, :]])
    # merge all small weight trajectories:
#    Wsmall = W[W < Wtot/steps/2]
    Wsum = 0
    init= 0 
    Itemp = np.array([], dtype = 'int')
    Wtemp = np.array([])
    conSmall = (W <= Wtot/steps/2)
    for i in range(conSmall.sum()): 
        Wsum = Wsum + W[i]
        if np.logical_or(Wsum > Wtot/steps, i+1 == conSmall.sum()): 
            P = np.cumsum(W[init:i+1])/W[init:i+1].sum()
            Itemp = np.append(Itemp, init + (P < np.random.rand(1) ).sum() )
            Wtemp = np.append(Wtemp, Wsum)
            init = i + 1 
            Wsum = 0
    W = np.hstack([Wtemp, W[~conSmall]])
    N = np.vstack([N[Itemp, :], N[~conSmall, :]])
    return N,  W


if __name__ == '__main__':
    # Parameters for the network to work
    start_time = time.time()
    N = 40 # number of nodes
    sims = 10 # Number of simulations at each step
    k = 5 # Average number of neighbors for each node
    x = 0.2 # intial infection percentage
    lam = 1.6 # The reproduction number
    Num_inf = int(x*N) # Number of initially infected nodes
    it = 500
    jump = 10
    Alpha = 1.0 # Recovery rate
    Beta_avg = Alpha * lam / k # Infection rate for each node
    epsilon = 0.0 # The normalized std (second moment divided by the first) of the network
    Beta = Beta_avg / (1 + epsilon ** 2) # This is so networks with different std will have the reproduction number

    # F0 = N*(1-dn)
    # Fpoint = root(RateEquation,  F0, args = (N, dn, mu))
    # FP = Fpoint.x # this is the unstable fixed point, over this point we switch.

    tau = 1/(Num_inf*Alpha+N*Beta*k)

    # creates a series of sims networks with Num_inf infections
    n = np.zeros((N, sims))
    for i in range(sims):
        for j in rand.sample(range(0, N - 1), Num_inf):
            n[j,i] = 1.0

    t = np.zeros(sims)
    weights = np.ones(sims)/sims
    n, weights, death = GillespieMC(n, weights, tau * 10, k, N, Alpha, Beta)

    Death = np.array([death])
    Nlimits = np.array([N+1, N*(1-1/lam), 0])

    n_min = n.min(0)
    Nlimits = np.insert(Nlimits, -2, n_min, axis = 0)
    WW = np.zeros(it)

    for j in range(it):
        n,  weights, death = GillespieMC(n, weights, tau * 10, k, N, Alpha, Beta)
        Death = np.append(Death, death)
        if np.logical_and(np.any(n.min(0) < n_min - jump), np.all(n.min(0) > FP)):
            n_min = n.min(0)
            Nlimits =np.insert(Nlimits, -2, n_min, axis = 0)

        Bins = (np.size(Nlimits, axis =0) -1)
        A = np.zeros([Bins, np.size(n,0)], dtype='bool')
        for q in range(Bins):
            A[q, :] = np.logical_and(np.sum(A[:q, :], 0) == 0, np.all(Nlimits[q+1, :]<=n, axis=1))
        WW[j] = weights[A[-1, :]].sum()

        # nf, wf = resample(n[A[0,:], :], weights[A[0, :]], steps)
        nf, wf = resample(n[A[0,:], :], weights[A[0, :]], sims)
        for q in range(1, Bins):
            if A[q,:].sum() != 0 :
                # n1, w1 = resample(n[A[q,:], :], weights[A[q, :]], steps)
                n1, w1 = resample(n[A[q,:], :], weights[A[q, :]], sims)
                nf = np.vstack([nf, n1])
                wf = np.hstack([wf, w1])
        n = nf
        weights = wf
        print(j)

    TAU = tau/np.mean((np.diff(WW[99:])+Death[101:]).reshape(int((it-100)/10), 10), axis = 1).mean()
    print('Check if probability is conserved: Weights + Death = ' ,weights.sum() + Death.sum())
    print('the time of switch is ', TAU)

    B = {}
    B['flux'] = Death
    B['weights'] = weights
    B['n'] = n
    B['tau'] = tau
    B['dn'] = dn
    B['N'] = N
    B['mu'] = mu
    B['TrajPerBin'] = sims


    dire = 'N_'+ np.str(N)  + '_dn_' + np.str(dn) + '_mu_'+ np.str(mu)
    savemat(dire,B)
    print("--- %s seconds ---" % (time.time() - start_time))


    
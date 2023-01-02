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
    G = nx.random_regular_graph(k,N)
    Adj = nx.adjacency_matrix(G,nodelist=range(0,N)) # Later this step should be removed appears only for debugging
    # Adj = nx.to_numpy_matrix(G) # Later this step should be removed appears only for debugging
    # Adj =nx.to_numpy_array(G)
    steps_c = np.size(n,1)
    t = np.zeros(steps_c)
    # ng = np.zeros([steps_c, sims])
    ng = np.zeros(np.shape(n))
    wg = np.zeros(steps_c)
    it = 10**6
    keep = it 
    j = 0 
    death = 0 
    # add_n = AddingMatrix(D)

    while 1 : 
        if keep >= it - np.size(n,0):
            r2 = np.random.rand(it)
            r1 = -np.log(np.random.rand(it))
            keep = 0 
        # a_vec = np.hstack([n*(n-1)*A, n+ n*(n-1)*(n-2)*B , np.repeat(mu*n, D-1, axis=1)])
        # a_vec = np.hstack([Beta * np.matmul(np.matmul(Adj,n), (np.ones(n.size)-n)),Alpha*n])
        # a_vec = np.hstack([Beta * Adj*n*(np.ones(n.shape)-n),Alpha*n])
        # a_vec = np.vstack([Beta * Adj*n*(np.ones(n.shape)-n),Alpha*n])
        a_vec = Beta * Adj*n*(np.ones(n.shape)-n)+Alpha*n
        # a = a_vec.sum(1)
        a = a_vec.sum(0)
        # index = (np.transpose(a_vec.cumsum(1))/a < r2[keep:keep+steps_c]).sum(0)
        # index = (np.transpose(a_vec.cumsum(0))/a < r2[keep:keep+steps_c]).sum(0)
        index = ((a_vec.cumsum(0))/a < r2[keep:keep+steps_c]).sum(0)
        t = t + 1/a*r1[keep:keep+steps_c]
        # n = n + add_n[index]
        for i in range(index.size): n[index[i], i] = 1 - n[index[i], i]
        # n = n + (1-2*n[index])
        # n = n + np.transpose(add_n[:, index])
        # keep = keep +  np.size(n, 0)
        keep = keep +  np.size(n, 1)
        if np.any(n.sum(0)<=0):
            con = n.sum(0)<=0
            y = con.sum()
            death = death + weights[con].sum()
            t = t[~con]
            n = n[~con,:]

            steps_c = np.size(n,1)
            weights = weights[~con]
            wg = wg[:-y]
            # ng = ng[:-y, :]
            ng = ng[:, :-y]
            if steps_c == 0:
               break 
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
#    N , M, dn, dm, mu, steps, it =  int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),int(sys.argv[4]), float(sys.argv[5]),  int(sys.argv[6])
    start_time = time.time()
    N = 40
    sims = 10
    k = 5
    x = 0.2
    lam = 1.6
    Num_inf = int(x*N)
    # steps = 1000
    it = 500
    jump = 10
    Alpha = 1.0
    Beta_avg = Alpha * lam / k
    epsilon = 0.0
    Beta = Beta_avg / (1 + epsilon ** 2)

# N = np.array([500, 500, 500])
    # dn = np.array([0.42, 0.38, 0.39])
    # mu = 4
    # print('mu = ' ,mu)
    # print('N = ' ,N)
    # print('dn = ' ,dn)
    # finding all permutations, which we treat as path to extinction
    # D = dn.size

    # F0 = N*(1-dn)
    # Fpoint = root(RateEquation,  F0, args = (N, dn, mu))
    # FP = Fpoint.x # this is the unstable fixed point, over this point we switch.

    # tau = 0.1* (1 - dn[0])/(2*dn[0])
    # tau = 0.1 * (1 - dn[0]) / (2 * dn[0])
    tau = 1/(Num_inf*Alpha+N*Beta*k)

    # n = np.ones([steps, D], dtype = 'int64')* N*(1+dn)
    n = np.zeros((N, sims))
    for i in range(sims):
        for j in rand.sample(range(0, N - 1), Num_inf):
            n[j,i] = 1.0

    # n = np.zeros((N, sims))

    # t = np.zeros(steps)
    # weights = np.ones(steps)/steps
    t = np.zeros(sims)
    weights = np.ones(sims)/sims
    # n,  weights, death = GillespieMC(n, weights ,  N, dn, mu, tau*10, D)
    n, weights, death = GillespieMC(n, weights, tau * 10, k, N, Alpha, Beta)

    Death = np.array([death])
    # Nlimits = np.vstack([10*N, FP, np.zeros(FP.size)])
    Nlimits = np.array([N+1, N*(1-1/lam), 0])

    n_min = n.min(0)
    Nlimits = np.insert(Nlimits, -2, n_min, axis = 0)
    WW = np.zeros(it)

    for j in range(it):
        # n,  weights, death = GillespieMC(n, weights ,  N, dn, mu, tau, D)
        n,  weights, death = GillespieMC(n, weights, tau * 10, k, N, Alpha, Beta)
        Death = np.append(Death, death)
        # if np.logical_and(np.any(n.min(0) < n_min - jump), np.all(n.min(0) > FP)):
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


    
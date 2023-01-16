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
#from scipy import linalg
from scipy.io import savemat
import sys 
from sympy.utilities.iterables import multiset_permutations
import time
import networkx as nx
import random as rand
import rand_networks


def GillespieMC(n, weights , tau, k, N, Alpha, Beta,G):
    steps_c = np.size(n,1) # Number of simulations
    t = np.zeros(steps_c)
    ng = np.zeros(np.shape(n))
    wg = np.zeros(steps_c)
    it = 10**6
    keep = it
    j = 0 
    death = 0 

    while 1 : 
        if keep >= it - np.size(n,1):
            r2 = np.random.rand(it)
            r1 = -np.log(np.random.rand(it))
            keep = 0 
        a_vec = Beta * nx.adjacency_matrix(G,nodelist=range(0,N))*n*(np.ones(n.shape)-n)+Alpha*n # Rates matrix each, columns different networks rows different nodes
        a = a_vec.sum(0)
        index = ((a_vec.cumsum(0))/a < r2[keep:keep+steps_c]).sum(0) # First index instance rate for a rate that's above the random number
        t = t + 1/a*r1[keep:keep+steps_c]
        index_arrange = np.arange(index.size)
        n[index[index_arrange],index_arrange] = 1 - n[index[index_arrange],index_arrange]
        # for i in range(index.size): n[index[i], i] = 1 - n[index[i], i] # Change the infect\heal nodes from index array
        keep = keep +  np.size(n, 1)
        # Finds out if a network reached extinction and if so remove it from the dynamical process
        if np.any(n.sum(0)<=0):
            con = n.sum(0)<=0
            y = con.sum()
            death = death + weights[con].sum()
            t = t[~con]
            n = n[:,~con]
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
    n = n.T
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
    return N.T,  W



def run_sim(N,sims,it,k,x,lam,jump,Alpha,Beta,network_number,tau,infile,mf_solution,new_trajcetory_bin):
    start_time = time.time()
    Num_inf = int(x*N) # Number of initially infected nodes
    # Parameters for the network to work
    # G = nx.complete_graph(N) # Creates a random graphs with k number of neighbors
    G = nx.read_gpickle(infile)
    relaxation_time  = 10
    # tau = 1/(Num_inf*Alpha+N*Beta*k)


    # creates a series of sims networks with Num_inf infections
    n = np.zeros((N, sims))
    for i in range(sims):
        for j in rand.sample(range(0, N - 1), Num_inf):
            n[j,i] = 1.0

    t = np.zeros(sims)
    weights = np.ones(sims)/sims
    # n, weights, death = GillespieMC(n, weights, tau , k, N, Alpha, Beta,G)

    Death = np.array([])
    # Nlimits = np.array([N+1, N*(1-1/lam),0])
    Nlimits = np.array([N+1, mf_solution,0])

    # n_min = n.min(0)
    # n_min = np.min(np.sum(n,axis=0))
    n_min = N*(1-1/lam)
    # Nlimits = np.insert(Nlimits, -1, n_min, axis = 0)

    for j in range(it):
        n,  weights, death = GillespieMC(n, weights, tau, k, N, Alpha, Beta,G)
        Death = np.append(Death, death)
        # n_min_new = np.quantile(np.sum(n, axis=0), 0.05)
        # n_min_new = np.partition(np.sum(n, axis=0), 5)[5]
        n_min_new = np.partition(np.sum(n, axis=0), new_trajcetory_bin)[new_trajcetory_bin]
        if n_min_new < n_min - jump and Nlimits[-2]>jump:
            # n_min = np.min(np.sum(n,axis=0))
            n_min = n_min_new
            Nlimits =np.insert(Nlimits, -1, n_min, axis = 0)

        Bins = np.size(Nlimits) -1
        A = np.zeros([Bins, np.size(n,1)], dtype='bool')
        for q in range(Bins):
            A[q, :] = np.logical_and(Nlimits[q+1] < np.sum(n,axis=0), Nlimits[q] >= np.sum(n,axis=0)) #np.logical_and(np.sum(A[:q, :], 0) == 0, np.all(Nlimits[q+1, :]<=n, axis=1))

        # nf, wf = resample(n[A[0,:], :], weights[A[0, :]], steps)
        nf, wf = resample(n[:,A[0,:]], weights[A[0, :]], sims)
        for q in range(1, Bins):
            if A[q,:].sum() != 0 :
                # n1, w1 = resample(n[A[q,:], :], weights[A[q, :]], steps)
                n1, w1 = resample(n[:, A[q,:]], weights[A[q, :]], sims)
                nf = np.hstack([nf, n1])
                wf = np.hstack([wf, w1])
        n = nf
        total_infected_for_sim = np.sum(n,axis=0)
        weights = wf
        print(j)

    TAU = tau/np.mean((Death[relaxation_time:]))  #.reshape(int((it-100)/10), 10), axis = 1).mean()
    theory_well_mixed_mte = (1/Alpha)*np.sqrt(2*np.pi/N)*(lam/(lam-1)**2)*np.exp(N*(np.log(lam)+1/lam-1))

    print('Check if probability is conserved: Weights + Death = ' ,weights.sum() + Death.sum())
    print('Sick {}, Healthy {}'.format(weights.sum(), Death.sum()))
    print('the time of switch is ', TAU)
    print('Theory and numeric ratio',TAU/theory_well_mixed_mte)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(Nlimits)
    print(Death)
    np.save('tau_' + str(network_number) + '.npy',TAU)
    np.save('Deaths_' + str(network_number) + '.npy',Death)
    np.save('Nlimits_'+ str(network_number) + '.npy',Nlimits)
    np.save('weights_'+ str(network_number) + '.npy',weights)
    np.save('total_infected_for_sim_' + str(network_number) + '.npy', total_infected_for_sim)


def act_as_main():
    N = 300 # number of nodes
    sims = 100 # Number of simulations at each step
    # k = 100 # Average number of neighbors for each node
    k = 100 # Average number of neighbors for each node

    x = 0.2 # intial infection percentage
    lam = 1.3 # The reproduction number
    it = 70
    jump = 10
    Alpha = 1.0 # Recovery rate
    Beta_avg = Alpha * lam / k # Infection rate for each node
    eps_din,eps_dout = 0.1,0.0 # The normalized std (second moment divided by the first) of the network
    Beta = Beta_avg / (1 + eps_din*eps_dout) # This is so networks with different std will have the reproduction number
    # G = nx.random_regular_graph(k,N) # Creates a random graphs with k number of neighbors
    network_num = 0
    tau = 1.0
    new_trajcetory_bin = 5
    d1_in, d1_out, d2_in, d2_out = int(k * (1 - eps_din)), int(k * (1 - eps_dout)), int(k * (1 + eps_din)), int(
        k * (1 + eps_dout))
    G = rand_networks.random_bimodal_directed_graph(d1_in, d1_out, d2_in, d2_out, N)
    infile = 'GNull_{}.pickle'.format(network_num)
    nx.write_gpickle(G, infile)
    # y1star=(-2*eps_din*(1 + eps_dout*eps_din)+ lam*(-1 + eps_din)*(1 + (-1 + 2*eps_dout)*eps_din)+ np.sqrt(lam**2 +eps_din*(4*eps_din +lam**2*eps_din*(-2 +eps_din**2) +4*eps_dout*(lam -(-2 + lam)*eps_din**2) +4*eps_dout**2*eps_din*(lam -(-1 + lam)*eps_din**2))))/(4*lam*(-1 +eps_dout)*(-1 +eps_din)*eps_din)
    # y2star=(lam + eps_din*(-2 + 2*lam +lam*eps_din+ 2*eps_dout*(lam +(-1 + lam)*eps_din)) -np.sqrt(lam**2 +eps_din*(4*eps_din +lam**2*eps_din*(-2 +eps_din**2) +4*eps_dout*(lam -(-2 + lam)*eps_din**2) +4*eps_dout**2*eps_din*(lam -(-1 + lam)*eps_din**2))))/(4*lam*(1 +eps_dout)*eps_din*(1 + eps_din))
    # Istar = y1star +y2star
    Istar = (1 - 1/lam) * N
    run_sim(N,sims,it,k,x,lam,jump,Alpha,Beta,network_num,tau,infile,Istar,new_trajcetory_bin)



if __name__ == '__main__':
    # act_as_main()
    if sys.argv[1] == 'bd' or sys.argv[1]=='gauss_c':
    # Run the extinction program for bimodal directed networks
        run_sim(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
        float(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8]),float(sys.argv[9]),float(sys.argv[10]),int(sys.argv[11]),
                float(sys.argv[12]),sys.argv[13],float(sys.argv[14]),int(sys.argv[15]))
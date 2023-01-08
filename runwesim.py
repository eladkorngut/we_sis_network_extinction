import numpy as np
import os

import we_sis_network_extinction

if __name__ == '__main__':
    # Parameters for the network to work
    N = 40 # number of nodes
    lam = 1.1 # The reproduction number
    number_of_networks = 1
    sims = 100 # Number of simulations at each step
    # k = 100 # Average number of neighbors for each node
    k = N # Average number of neighbors for each node
    x = 0.2 # intial infection percentage
    Num_inf = int(x*N) # Number of initially infected nodes
    it = 20
    jump = 10
    Alpha = 1.0 # Recovery rate
    Beta_avg = Alpha * lam / k # Infection rate for each node
    epsilon = 0.0 # The normalized std (second moment divided by the first) of the network
    Beta = Beta_avg / (1 + epsilon ** 2) # This is so networks with different std will have the reproduction number
    # G = nx.random_regular_graph(k,N) # Creates a random graphs with k number of neighbors
    relaxation_time  = 10
    # tau = 1/(Num_inf*Alpha+N*Beta*k)
    tau = 1.0
    parameters =np.array([N,sims,it,k,x,lam,jump,Num_inf,number_of_networks,tau])

    foldername = 'temp_test_weight_numinf'
    os.mkdir(foldername)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(foldername)
    np.save('parameters.npy',parameters)
    for i in range(number_of_networks):
        we_sis_network_extinction.run_sim(N,sims,it,k,x,lam,jump,Alpha,Beta,i,tau)

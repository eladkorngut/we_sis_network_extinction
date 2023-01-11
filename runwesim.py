import numpy as np
import os
import we_sis_network_extinction
import rand_networks
import networkx as nx

def act_as_main(foldername,parameters,number_of_networks,d1_in, d1_out, d2_in, d2_out, N,sims,it,k,x,lam,jump,Alpha,Beta,tau,Istar,new_trajcetory_bin):
    # This program will run the we_sis_network_extinction.py on the laptop\desktop
    os.mkdir(foldername)
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(foldername)
    np.save('parameters.npy',parameters)
    for i in range(number_of_networks):
        G = rand_networks.random_bimodal_directed_graph(d1_in, d1_out, d2_in, d2_out, N)
        infile = 'GNull_{}.pickle'.format(i)
        nx.write_gpickle(G, infile)
        we_sis_network_extinction.run_sim(N,sims,it,k,x,lam,jump,Alpha,Beta,i,tau,infile,Istar,new_trajcetory_bin)

def job_to_cluster(foldername,parameters,number_of_networks,d1_in, d1_out, d2_in, d2_out, N,Beta,tau,Istar,it,new_trajcetory_bin):
    # This function submit jobs to the cluster with the following program keys:
    # bd: creates a bimodal directed networks and find its mean time to extinction
    prog = 'bd'
    os.mkdir(foldername)
    os.chdir(foldername)
    np.save('parameters.npy',parameters)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for i in range(number_of_networks):
        G = rand_networks.random_bimodal_directed_graph(d1_in, d1_out, d2_in, d2_out, N)
        # G = nx.complete_graph(N)
        infile = 'GNull_{}.pickle'.format(i)
        nx.write_gpickle(G, infile)
        os.system(dir_path + '/slurm.serjob python3 ' + dir_path + '/we_sis_network_extinction.py ' + str(prog) + ' ' +
        str(N) + ' ' + str(sims) + ' ' + str(it) + ' ' + str(k) + ' ' + str(x) + ' ' + str(lam) + ' ' + str(jump) + ' ' + str(Alpha) + ' ' + str(Beta) +
        ' ' + str(i) + ' ' + str(tau)+ ' ' + str(infile)+ ' ' + str(Istar)+ ' ' + str(new_trajcetory_bin))


if __name__ == '__main__':
    # Parameters for the network to work
    N = 350 # number of nodes
    lam = 1.5 # The reproduction number
    number_of_networks = 10
    sims = 1000 # Number of simulations at each step
    # k = N # Average number of neighbors for each node
    k = 50 # Average number of neighbors for each node
    x = 0.2 # intial infection percentage
    Num_inf = int(x*N) # Number of initially infected nodes
    it = 100
    jump = 5
    Alpha = 1.0 # Recovery rate
    Beta_avg = Alpha * lam / k # Infection rate for each node
    eps_din,eps_dout = 0.0,0.0 # The normalized std (second moment divided by the first) of the network
    Beta = Beta_avg / (1 + eps_din*eps_dout) # This is so networks with different std will have the reproduction number
    # G = nx.random_regular_graph(k,N) # Creates a random graphs with k number of neighbors
    relaxation_time  = 10
    # tau = 1/(Num_inf*Alpha+N*Beta*k)
    tau = 1.0
    new_trajcetory_bin = 5

    parameters = np.array([N,sims,it,k,x,lam,jump,Num_inf,number_of_networks,tau,eps_din,eps_dout])
    graphname  = 'GNull'
    foldername = 'wellmixed_N100_lam16_tau1_it100_jump5_quant5_sims1000_net10_eps0'
    d1_in, d1_out, d2_in, d2_out = int(k * (1 - eps_din)), int(k * (1 - eps_dout)), int(k * (1 + eps_din)), int(
        k * (1 + eps_dout))
    # y1star=(-2*eps_din*(1 + eps_dout*eps_din)+ lam*(-1 + eps_din)*(1 + (-1 + 2*eps_dout)*eps_din)+ np.sqrt(lam**2 +eps_din*(4*eps_din +lam**2*eps_din*(-2 +eps_din**2) +4*eps_dout*(lam -(-2 + lam)*eps_din**2) +4*eps_dout**2*eps_din*(lam -(-1 + lam)*eps_din**2))))/(4*lam*(-1 +eps_dout)*(-1 +eps_din)*eps_din)
    # y2star=(lam + eps_din*(-2 + 2*lam +lam*eps_din+ 2*eps_dout*(lam +(-1 + lam)*eps_din)) -np.sqrt(lam**2 +eps_din*(4*eps_din +lam**2*eps_din*(-2 +eps_din**2) +4*eps_dout*(lam -(-2 + lam)*eps_din**2) +4*eps_dout**2*eps_din*(lam -(-1 + lam)*eps_din**2))))/(4*lam*(1 +eps_dout)*eps_din*(1 + eps_din))
    # Istar = (y1star +y2star)*N
    Istar = (1 - 1/lam) * N


    # What's the job to run either on the cluster or on the laptop
    job_to_cluster(foldername, parameters, number_of_networks, d1_in, d1_out, d2_in, d2_out, N, Beta, tau, Istar,it,new_trajcetory_bin)

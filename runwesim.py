import numpy as np
import os
import we_sis_network_extinction
import rand_networks
import networkx as nx

def act_as_main(foldername,parameters,Istar):
    # This program will run the we_sis_network_extinction.py on the laptop\desktop
    os.mkdir(foldername)
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(foldername)
    N, sims, it, k, x, lam, jump, Num_inf, number_of_networks, tau, eps_din, eps_dout,new_trajcetory_bin,prog,Beta_avg = parameters
    if prog == 'bd':
        np.save('parameters.npy',parameters)
        # G = nx.complete_graph(N)
        d1_in, d1_out, d2_in, d2_out = int(k * (1 - eps_din)), int(k * (1 - eps_dout)), int(k * (1 + eps_din)), int(
            k * (1 + eps_dout))
        Beta = Beta_avg / (1 + eps_din * eps_dout)  # This is so networks with different std will have the reproduction number
    for i in range(int(number_of_networks)):
        # G = rand_networks.random_bimodal_directed_graph(d1_in, d1_out, d2_in, d2_out, N)
        if prog=='bd':
            G = rand_networks.random_bimodal_directed_graph(int(d1_in), int(d1_out), int(d2_in), int(d2_out), int(N))
        else:
            G = rand_networks.configuration_model_directed_graph(prog, float(eps_din), float(eps_dout), int(k), int(N))
            k_avg_graph = np.mean([G.in_degree(n) for n in G.nodes()])
            Beta_graph = float(lam)/k_avg_graph
            eps_in_graph = np.std([G.in_degree(n) for n in G.nodes()])/k_avg_graph
            eps_out_graph = np.std([G.out_degree(n) for n in G.nodes()])/k_avg_graph
            Beta = Beta_graph / (1 + np.sign(float(eps_din))*eps_in_graph * np.sign(float(eps_dout))* eps_out_graph)
            parameters = np.array([N, sims, it, k, x, lam, jump, Num_inf, number_of_networks, tau, eps_in_graph,
                                   eps_out_graph, new_trajcetory_bin,prog, Beta])
            np.save('parameters_{}.npy'.format(i), parameters)
        infile = 'GNull_{}.pickle'.format(i)
        nx.write_gpickle(G, infile)
        we_sis_network_extinction.run_sim(int(N),int(sims),int(it),int(k),float(x),float(lam),int(jump),float(Alpha),
                                          float(Beta),int(i),float(tau),infile,float(Istar),int(new_trajcetory_bin))

def job_to_cluster(foldername,parameters,Istar):
    # This function submit jobs to the cluster with the following program keys:
    # bd: creates a bimodal directed networks and find its mean time to extinction
    os.mkdir(foldername)
    os.chdir(foldername)
    N, sims, it, k, x, lam, jump, Num_inf, number_of_networks, tau, eps_din, eps_dout,new_trajcetory_bin,prog,Beta_avg = parameters
    if prog == 'bd':
        np.save('parameters.npy',parameters)
        # G = nx.complete_graph(N)
        d1_in, d1_out, d2_in, d2_out = int(k * (1 - eps_din)), int(k * (1 - eps_dout)), int(k * (1 + eps_din)), int(
            k * (1 + eps_dout))
        Beta = Beta_avg / (1 + eps_din * eps_dout)  # This is so networks with different std will have the reproduction number
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for i in range(int(number_of_networks)):
        if prog=='bd':
            G = rand_networks.random_bimodal_directed_graph(int(d1_in), int(d1_out), int(d2_in), int(d2_out), int(N))
        else:
            G = rand_networks.configuration_model_directed_graph(prog, float(eps_din), float(eps_dout), int(k), int(N))
            k_avg_graph = np.mean([G.in_degree(n) for n in G.nodes()])
            Beta_graph = float(lam)/k_avg_graph
            eps_in_graph = np.std([G.in_degree(n) for n in G.nodes()])/k_avg_graph
            eps_out_graph = np.std([G.out_degree(n) for n in G.nodes()])/k_avg_graph
            Beta = Beta_graph / (1 + np.sign(float(eps_din))*eps_in_graph * np.sign(float(eps_dout))* eps_out_graph)
            parameters = np.array([N, sims, it, k, x, lam, jump, Num_inf, number_of_networks, tau, eps_in_graph,
                                   eps_out_graph, new_trajcetory_bin,prog, Beta])
            np.save('parameters_{}.npy'.format(i), parameters)
        infile = 'GNull_{}.pickle'.format(i)
        nx.write_gpickle(G, infile)
        os.system(dir_path + '/slurm.serjob python3 ' + dir_path + '/we_sis_network_extinction.py ' + str(prog) + ' ' +
        str(N) + ' ' + str(sims) + ' ' + str(it) + ' ' + str(k) + ' ' + str(x) + ' ' + str(lam) + ' ' + str(jump) + ' ' + str(Alpha) + ' ' + str(Beta) +
        ' ' + str(i) + ' ' + str(tau)+ ' ' + str(infile)+ ' ' + str(Istar)+ ' ' + str(new_trajcetory_bin))


if __name__ == '__main__':
    # Parameters for the network to work
    N = 400 # number of nodes
    lam = 1.4 # The reproduction number
    number_of_networks = 10
    sims = 2000 # Number of simulations at each step
    # k = N # Average number of neighbors for each node
    k = 100 # Average number of neighbors for each node
    x = 0.2 # intial infection percentage
    Num_inf = int(x*N) # Number of initially infected nodes
    it = 100
    jump = 5
    Alpha = 1.0 # Recovery rate
    Beta_avg = Alpha * lam / k # Infection rate for each node
    eps_din,eps_dout = 0.1,0.0 # The normalized std (second moment divided by the first) of the network
    # G = nx.random_regular_graph(k,N) # Creates a random graphs with k number of neighbors
    relaxation_time  = 10
    # tau = 1/(Num_inf*Alpha+N*Beta*k)
    tau = 1.0
    new_trajcetory_bin = 5
    prog = 'gauss_c'

    parameters = np.array([N,sims,it,k,x,lam,jump,Num_inf,number_of_networks,tau,eps_din,eps_dout,new_trajcetory_bin,prog,Beta_avg])
    graphname  = 'GNull'
    foldername = 'gauss_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin01_epsout0'
    # y1star=(-2*eps_din*(1 + eps_dout*eps_din)+ lam*(-1 + eps_din)*(1 + (-1 + 2*eps_dout)*eps_din)+ np.sqrt(lam**2 +eps_din*(4*eps_din +lam**2*eps_din*(-2 +eps_din**2) +4*eps_dout*(lam -(-2 + lam)*eps_din**2) +4*eps_dout**2*eps_din*(lam -(-1 + lam)*eps_din**2))))/(4*lam*(-1 +eps_dout)*(-1 +eps_din)*eps_din)
    # y2star=(lam + eps_din*(-2 + 2*lam +lam*eps_din+ 2*eps_dout*(lam +(-1 + lam)*eps_din)) -np.sqrt(lam**2 +eps_din*(4*eps_din +lam**2*eps_din*(-2 +eps_din**2) +4*eps_dout*(lam -(-2 + lam)*eps_din**2) +4*eps_dout**2*eps_din*(lam -(-1 + lam)*eps_din**2))))/(4*lam*(1 +eps_dout)*eps_din*(1 + eps_din))
    # Istar = (y1star +y2star)*N
    Istar = (1 - 1/lam) * N


    # What's the job to run either on the cluster or on the laptop
    job_to_cluster(foldername,parameters,Istar)
    # act_as_main(foldername,parameters,Istar)

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import expon
import scipy.io as sio
import networkx as nx
from scipy.optimize import fsolve


def plot_detahs_v_it(filename,directory_name):
    net_number = 0
    # Deaths= [None]*number_of_networks
    fig_death, ax_death = plt.subplots()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for f in range(len(filename)):
        os.chdir(dir_path+directory_name)
        os.chdir(filename[f])
        N, sims, it, k, x, lam, jump, Num_inf, number_of_networks, tau, eps_din, eps_dout,new_trajcetory_bin,prog,Beta_avg,skewness_in,skewness_out = np.load('parameters.npy')
        Deaths = np.load('Deaths_{}.npy'.format(net_number))
    # ax_death.semilogy(np.arange(it), Deaths)
    ax_death.semilogy(Deaths)
    plt.xlabel('it')
    plt.ylabel('Deaths')
    plt.title('Deaths vs it N={}, R0={}'.format(N,lam))
    os.chdir(dir_path)
    fig_death.savefig('Death_v_it.png',dpi=200)
    plt.show()

def plot_theory_well_mixed(lam_net,extinction_time_mean,extinction_time_std,N):
    lam_theory = np.linspace(np.min(lam_net),np.max(lam_net),100)
    # N_theory = np.linspace(np.min(N), np.max(N), 100)
    # theory_well_mixed_mte = (1/Alpha)*np.sqrt(2*np.pi/N_theory)*(lam_net/(lam_net-1)**2)*np.exp(N_theory*(np.log(lam_net)+1/lam_net-1))
    theory_well_mixed_mte = (1/Alpha)*np.sqrt(2*np.pi/N)*(lam_theory/(lam_theory-1)**2)*np.exp(N*(np.log(lam_theory)+1/lam_theory-1))
    fig_extinction, ax_extinction = plt.subplots()
    # plt.scatter(N_net,np.log(extinction_time_mean))
    # plt.plot(N_theory,np.log(theory_well_mixed_mte))
    plt.errorbar(lam_net,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o')
    # plt.errorbar(N,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o')
    plt.plot(lam_theory,np.log(theory_well_mixed_mte))
    # plt.plot(N_theory,np.log(theory_well_mixed_mte))
    # plt.xlabel('N')
    plt.xlabel('R')
    plt.ylabel('ln(T)')
    plt.title('MTE vs R N={}'.format(N))
    # plt.title('N vs ln(T) N={}'.format(int(N)))
    # fig_extinction.savefig('extinction_v_N.png',dpi=200)
    fig_extinction.savefig('extinction_v_lam.png',dpi=200)
    plt.show()


def plot_theory_undirected(lam,extinction_time_mean,extinction_time_std,N,eps_din,eps_dout):
    # eps_theory = np.linspace(np.min(eps_din),np.max(eps_din),100)
    # s0 = np.log(lam)+1/lam-1
    offset = 0
    # f_R0 = -((lam-1)*(1-12*lam+3*lam**2)+8*lam**2*np.log(lam))/(4*lam**3)
    # theory_mj_mte = N*offset - N*f_R0*eps_theory**2
    fig_extinction, ax_extinction = plt.subplots()
    mat_master_eq = sio.loadmat('bimodal_mte.mat')
    tau_master = mat_master_eq['tau'][0]
    epsilon_mu_master = mat_master_eq['epsilon_mu'][0]
    ax_extinction.plot(epsilon_mu_master,np.log(tau_master),linewidth=2,label='Master',linestyle='-',color='b')
    ax_extinction.errorbar(eps_din,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o',label='WE',color='r')
    # plt.scatter(eps_din**2, np.log(extinction_time_mean))
    # plt.plot(eps_theory**2,theory_mj_mte)

    ax_extinction.set_xlabel(r'$\epsilon$')
    ax_extinction.set_ylabel('ln(T)')
    ax_extinction.set_title(r'Undirected Graph ln(T) vs $\epsilon$ N={} and R={}'.format(int(N),lam))
    plt.legend()
    fig_extinction.savefig('extinction_v_epsilon_undirected.png',dpi=200)
    plt.show()

def plot_theory_clancy(lam,extinction_time_mean,extinction_time_std,N,eps_din,eps_dout):
    # eps_theory = np.linspace(np.min(eps_din),np.max(eps_din),100)
    fig_extinction, ax_extinction = plt.subplots()
    epsilon_mu_theory = np.linspace(0.0, 0.99, 100)
    epsilon_lam =0.0
    offset_theory = -0.24
    offset_master = 0.5
    mat_master_eq = sio.loadmat('bimodal_mte.mat')
    tau_master = mat_master_eq['tau'][0]
    epsilon_mu_master = mat_master_eq['epsilon_mu'][0]
    zeta = lam / (2 * (1 + epsilon_lam * epsilon_mu_theory)) - 1/(1 - epsilon_mu_theory**2)
    D = zeta + np.sqrt(zeta**2 + (lam - 1) / (1 - epsilon_mu_theory**2))
    action_clancy = (1 / 2) * (np.log(1 + (1 - epsilon_mu_theory)* D) + np.log(1 + (1 + epsilon_mu_theory)* D)) - (
                D/ lam)
    ax_extinction.plot(epsilon_mu_master,np.log(tau_master)+offset_theory,linewidth=2,label='Master',linestyle='--',color='k')
    ax_extinction.errorbar(eps_din,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o',color='r',label='WE')

    # plt.scatter(eps_din**2, np.log(extinction_time_mean))
    ax_extinction.plot(epsilon_mu_theory,N*action_clancy+offset_master,color='b',label='Theory')
    ax_extinction.set_xlabel(r'$\epsilon$')
    ax_extinction.set_ylabel('ln(T)')
    plt.legend()
    ax_extinction.set_title(r'Pratialy heterogenous (Bimodal) ln(T) vs $\epsilon$ N={} and R={}'.format(int(N),lam))
    fig_extinction.savefig('extinction_v_epsilon_clancy.png',dpi=200)
    plt.show()

# def plot_theory_generic(lam,din,dout,k_avg_graph,eps_din_vec,eps_dout_vec,N):
#     fig_extinction, ax_extinction = plt.subplots()
#     for degree_in,degree_out,k0 in zip(din,dout,k_avg_graph):
#         dbig = fsolve(lambda dbig: (lam / N) * np.dot(degree_in / (k0 * np.ones(N) + dbig * degree_in), degree_out) - 1, 0.0)[0]
#         action = (1/N) * np.sum(np.log(np.ones(N) + din * dbig)) - (1 / lam) * dbig)
#         # ystari = (din * dbig) / (N * (k_avg_graph * np.ones(N) + din * dbig))
#         # Istar.apppend(N * np.sum(ystari))
#         for f in range(len(filename)):
#         os.chdir(dir_path+directory_name)
#         os.chdir(filename[f])
#         for i in range(int(number_of_networks)):
#             infile = 'GNull_{}.pickle'.format(i)
#             G = nx.read_gpickle(infile)
#             din = np.array([G.in_degree(i) for i in range(N)])
#             dout = np.array([G.out_degree(i) for i in range(N)])
#             k_avg_graph = np.mean([G.in_degree(n) for n in G.nodes()])
#             eps_in_graph.append(np.std(din)/k_avg_graph)
#             eps_out_graph.append(np.std(dout)/k_avg_graph)
#             dbig = fsolve(lambda dbig: (Beta / N) * np.dot(din / (k_avg_graph * np.ones(N) + dbig * din), dout) - 1, 0.0)[0]
#             ystari = (din * dbig) / (N * (k_avg_graph * np.ones(N) + din * dbig))
#             Istar.apppend(N * np.sum(ystari))
#             action.append((1/N)*np.sum(np.log(np.ones(N)+din*dbig))-(1/Beta)*dbig)
#     epsilon_lam =0.0
#     offset_theory = 0.0
#     offset_master = 0.0
#     zeta = lam / (2 * (1 + epsilon_lam * epsilon_mu_theory)) - 1/(1 - epsilon_mu_theory**2)
#     D = zeta + np.sqrt(zeta**2 + (lam - 1) / (1 - epsilon_mu_theory**2))
#     action_clancy = (1 / 2) * (np.log(1 + (1 - epsilon_mu_theory)* D) + np.log(1 + (1 + epsilon_mu_theory)* D)) - (
#                 D/ lam)
#     ax_extinction.plot(epsilon_mu_master,np.log(tau_master)+offset_theory,linewidth=2,label='Master',linestyle='--',color='k')
#     # ax_extinction.errorbar(eps_in_graph,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o',color='r',label='WE')
#     ax_extinction.errorbar(eps_in_graph,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o',color='r',label='WE')
#     ax_extinction.plot(eps_in_graph,N*action+offset_master,color='b',label='Theory')
#
#     # plt.scatter(eps_din**2, np.log(extinction_time_mean))
#     # ax_extinction.plot(epsilon_mu_theory,N*action_clancy+offset_master,color='b',label='Theory')
#     ax_extinction.set_xlabel(r'$\epsilon$')
#     ax_extinction.set_ylabel('ln(T)')
#     plt.legend()
#     ax_extinction.set_title(r'Pratialy heterogenous (Bimodal) ln(T) vs $\epsilon$ N={} and R={}'.format(int(N),lam))
#     fig_extinction.savefig('extinction_v_epsilon_clancy.png',dpi=200)
#     plt.show()


def plot_MTE(filename,directory_name,relaxation_time):
    Alpha =1.0
    extinction_time_mean, extinction_time_std,N_net,lam_net = np.empty(len(filename)),np.empty(len(filename)),np.empty(len(filename)),np.empty(len(filename))
    # extinction_time_mean, extinction_time_std,lam_net = np.empty(len(filename)),np.empty(len(filename)),np.empty(len(filename))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    eps_din_vec, eps_dout_vec = np.empty(int(len(filename))), np.empty(int(len(filename)))
    for f in range(len(filename)):
        os.chdir(dir_path+directory_name)
        os.chdir(filename[f])
        # N, sims, it, k, x, lam, jump, Num_inf, number_of_networks, tau, eps_din, eps_dout,new_trajcetory_bin,prog,Beta_avg,skewness_in,skewness_out = np.load('parameters.npy')
        N, sims, it, k, x, lam, jump, Num_inf, number_of_networks, tau, eps_din, eps_dout,new_trajcetory_bin,prog,Beta_avg,skewness_in,skewness_out = np.load('parameters_0.npy')
        # eps_din_vec[f] = eps_din
        # eps_dout_vec[f] =eps_dout
        extinction_time,eps_in_current,eps_out_current,dbig_net,action_theory =  np.empty(int(number_of_networks)),np.empty(int(number_of_networks)),np.empty(int(number_of_networks)),np.empty(int(number_of_networks)),np.empty(int(number_of_networks))
        for i in range(int(number_of_networks)):
            Death = np.load('Deaths_{}.npy'.format(i))
            # G = nx.read_gpickle('GNull_{}.pickle'.format(i))
            din= np.array([G.in_degree(i) for i in range(N)])
            dout = np.array([G.out_degree(i) for i in range(N)])
            k_avg_graph = np.mean([G.in_degree(n) for n in G.nodes()])
            dbig_net[i] = fsolve(lambda dbig: (lam / N) * np.dot(din / (k_avg_graph * np.ones(N) + dbig * din), dout) - 1,0.0)[0]
            N, sims, it, k, x, lam, jump, Num_inf, number_of_networks, tau, eps_din, eps_dout,new_trajcetory_bin,prog,Beta_avg = np.load('parameters_{}.npy'.format(i))
            # N, sims, it, k, x, lam, jump, Num_inf, number_of_networks, tau, eps_din, eps_dout,new_trajcetory_bin,prog,Beta_avg,skewness_in,skewness_out = np.load('parameters.npy')
            N = int(N)
            tau = float(tau)
            eps_in_current[i] = float(eps_din)
            eps_out_current[i] = float(eps_dout)
            extinction_time[i] = tau/np.mean((Death[relaxation_time:]))
            action_theory[i] = (1/N)*np.sum(np.log(np.ones(N)+din*dbig_net))-(1/lam)*dbig_net
        extinction_time_mean[f] = np.mean(extinction_time)
        extinction_time_std[f] =  np.std(extinction_time)
        eps_din_vec[f] = np.mean(eps_in_current)
        eps_dout_vec[f] = np.mean(eps_out_current)

        lam_net[f] = lam
        # N_net[f] = N
    os.chdir(dir_path)
    offset_master = 0.0
    fig_extinction, ax_extinction = plt.subplots()
    # ax_extinction.plot(eps_din_vec,np.log(tau_master)+offset_theory,linewidth=2,label='Master',linestyle='--',color='k')
    # ax_extinction.errorbar(eps_in_graph,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o',color='r',label='WE')
    ax_extinction.errorbar(eps_din_vec,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o',color='r',label='WE')
    ax_extinction.plot(eps_din,N*action_theory+offset_master,color='b',label='Theory')

    # plt.scatter(eps_din**2, np.log(extinction_time_mean))
    # ax_extinction.plot(epsilon_mu_theory,N*action_clancy+offset_master,color='b',label='Theory')
    ax_extinction.set_xlabel(r'$\epsilon$')
    ax_extinction.set_ylabel('ln(T)')
    plt.legend()
    ax_extinction.set_title(r'Pratialy heterogenous (Bimodal) ln(T) vs $\epsilon$ N={} and R={}'.format(int(N),lam))
    fig_extinction.savefig('extinction_v_epsilon_clancy.png',dpi=200)
    plt.show()

    # plot_theory_well_mixed(lam, extinction_time_mean, extinction_time_std, N_net)
    # plot_theory_well_mixed(lam_net, extinction_time_mean, extinction_time_std, N)
    # plot_theory_clancy(float(lam), extinction_time_mean, extinction_time_std, int(N),eps_din_vec,eps_dout_vec)
    # plot_theory_generic(lam,din,dout,k_avg_graph,eps_din_vec,eps_dout_vec)
    # plot_theory_undirected(lam, extinction_time_mean, extinction_time_std, N, eps_din_vec, eps_dout_vec)

    # plot_theory_undirected_weak_hetro(lam, extinction_time_mean, extinction_time_std, N,eps_din_vec,eps_dout_vec)

def plot_weight(filename,directory_name):
    net_number = 0
    fig_extinction, ax_extinction = plt.subplots()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for f in range(len(filename)):
        os.chdir(dir_path+directory_name)
        os.chdir(filename[f])
        N, sims, it, k, x, lam, jump, Num_inf, number_of_networks, tau, eps_din, eps_dout,new_trajcetory_bin,prog,Beta_avg = np.load('parameters.npy')
        weight = np.load('weights_{}.npy'.format(net_number))
        Nlimits = np.load('Nlimits_{}.npy'.format(net_number))
        total_infected_for_sim = np.load('total_infected_for_sim_{}.npy'.format(net_number))
        # plt.scatter(range(len(weight)), weight)
        hist, edges = np.histogram(total_infected_for_sim, bins=np.sort(Nlimits), weights=weight,density=True)
        hist1, edges1 = np.histogram(total_infected_for_sim, bins=20, weights=weight,density=True)
        # for i in range(int(number_of_networks)):
        #     weight = np.load('weights_{}_{}.npy'.format(i))
        #     plt.scatter(range(len(weight)), weight)
    # plt.bar(edges[:-1], hist, width=0.5, align='edge')
    plt.semilogy(edges[:-1], hist, '-o')
    plt.semilogy(edges1[:-1], hist1, '-o')
    os.chdir(dir_path)
    plt.xlabel('Sims')
    plt.ylabel('Probability')
    plt.title('Probability vs Sims N={}, R0={}'.format(N,lam))
    fig_extinction.savefig('Weight_v_Sims_N{}_R{}.png'.format(N,lam),dpi=200)
    plt.show()




if __name__ == '__main__':
    # Plot graphs of the we simulations
    # directory_name ='/analysis/clancy_case/diff_eps_mu/N400_lam14_diff_eps_clancy_case/'
    # directory_name ='/analysis/bimodal_undirected_network/N400_lam14_diff_eps_high_jumps/'
    # directory_name ='/analysis/gamma/diff_eps/clancy_case/diff_eps/N400_lam14/'
    # directory_name ='/analysis/clancy_case/diff_eps_mu/N1000_lam122_diff_eps_k200/new_results/new_results/'
    directory_name ='/analysis/lognorm/diff_eps/clancy_case/N1000_lam122_k40_it100_jump10_sims2000/'

    # filename = ['gauss_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin01_epsout0','gauss_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin02_epsout0','gauss_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin03_epsout0',
    #             'gauss_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin04_epsout0','gauss_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin05_epsout0','gauss_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin06_epsout0',
    #             'gauss_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin07_epsout0','gauss_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin084_epsout0']
    # filename = ['gamma_N400_k100_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin01_epsout0','gamma_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin02_epsout0',
    #             'gamma_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin03_epsout0','gamma_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin04_epsout0',
    #             'gamma_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin05_epsout0','gamma_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin06_epsout0',
    #             'gamma_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin07_epsout0','gamma_N400_lam14_tau1_it100_jump5_quant5_sims2000_net10_epsin084_epsout0']
    # filename = ['bimodal_N1000_k200_lam122_tau1_it100_jump10_quant5_sims500_net10_epsin01_epsout0','bimodal_N1000_k200_lam122_tau1_it100_jump10_quant5_sims500_net10_epsin02_epsout0',
    #             'bimodal_N1000_k200_lam122_tau1_it100_jump10_quant5_sims500_net10_epsin03_epsout0','bimodal_N1000_k200_lam122_tau1_it100_jump10_quant5_sims2000_net10_epsin04_epsout0',
    #             'bimodal_N1000_k200_lam122_tau1_it100_jump10_quant5_sims2000_net10_epsin05_epsout0',]
    # filename = ['bimodal_N1000_k200_lam122_tau1_it100_jump10_quant5_sims5000_net10_epsin07_epsout0']
    filename = ['lognorm_N1000_k40_lam122_tau1_it100_jump10_quant5_sims2000_net10_epsin01_epsout0','lognorm_N1000_k40_lam122_tau1_it100_jump10_quant5_sims2000_net10_epsin02_epsout0',
                'lognorm_N1000_k40_lam122_tau1_it100_jump10_quant5_sims2000_net10_epsin03_epsout0','lognorm_N1000_k40_lam122_tau1_it100_jump10_quant5_sims2000_net10_epsin04_epsout0',
                'lognorm_N1000_k40_lam122_tau1_it100_jump10_quant5_sims2000_net10_epsin05_epsout0','lognorm_N1000_k40_lam122_tau1_it100_jump10_quant5_sims2000_net10_epsin07_epsout0',
                'lognorm_N1000_k40_lam122_tau1_it100_jump10_quant5_sims2000_net10_epsin09_epsout0','lognorm_N1000_k40_lam122_tau1_it100_jump10_quant5_sims2000_net10_epsin10_epsout0',
                'lognorm_N1000_k40_lam122_tau1_it100_jump10_quant5_sims2000_net10_epsin20_epsout0']


    relaxation_time  = 20
    Alpha = 1.0
    plot_MTE(filename,directory_name,relaxation_time)
    # plot_weight(filename,directory_name)
    # plot_detahs_v_it(filename, directory_name)
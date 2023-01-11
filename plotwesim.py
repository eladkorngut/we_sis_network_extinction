import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import expon

def plot_detahs_v_it(number_of_networks,it):
    Deaths= [None]*number_of_networks
    for i in range(number_of_networks):
        Deaths[i] = np.load('Deaths_{}.npy'.format(i))
    fig_deaths, ax_deaths = plt.subplots()
    for d in Deaths:
        ax_deaths.semilogy(np.arange(it), Deaths[d])
    plt.xlabel('it')
    plt.ylabel('Deaths')
    plt.title('Deaths vs it N={}, R0={}'.format(N,lam))
    fig_deaths.savefig('Death_v_it.png',dpi=200)
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


def plot_theory_undirected_weak_hetro(lam,extinction_time_mean,extinction_time_std,N,eps_din,eps_dout):
    eps_theory = np.linspace(np.min(eps_din),np.max(eps_din),100)
    s0 = np.log(lam)+1/lam-1
    offset = s0
    f_R0 = -((lam-1)*(1-12*lam+3*lam**2)+8*lam**2*np.log(lam))/(4*lam**3)
    theory_mj_mte = N*offset - N*f_R0*eps_theory**2
    fig_extinction, ax_extinction = plt.subplots()
    # plt.errorbar(eps_din**2,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o')
    plt.scatter(eps_din**2, np.log(extinction_time_mean))
    plt.plot(eps_theory**2,theory_mj_mte)
    plt.xlabel('\epsilon**2')
    plt.ylabel('ln(T)')
    plt.title('ln(T) vs \epsilon**2 N={}'.format(int(N)))
    fig_extinction.savefig('extinction_v_epsilon.png',dpi=200)
    plt.show()


def plot_MTE(filename,directory_name,relaxation_time):
    Alpha =1.0
    extinction_time_mean, extinction_time_std,N_net,lam_net = np.empty(len(filename)),np.empty(len(filename)),np.empty(len(filename)),np.empty(len(filename))
    # extinction_time_mean, extinction_time_std,lam_net = np.empty(len(filename)),np.empty(len(filename)),np.empty(len(filename))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    eps_din_vec, eps_dout_vec = np.empty(int(len(filename))), np.empty(int(len(filename)))
    for f in range(len(filename)):
        os.chdir(dir_path+directory_name)
        os.chdir(filename[f])
        N, sims, it, k, x, lam, jump, Num_inf, number_of_networks,tau,eps_din,eps_dout = np.load('parameters.npy')

        eps_din_vec[f] =eps_din
        eps_dout_vec[f] =eps_dout
        extinction_time =  np.empty(int(number_of_networks))
        for i in range(int(number_of_networks)):
            Death = np.load('Deaths_{}.npy'.format(i))
            extinction_time[i] = tau/np.mean((Death[relaxation_time:]))
        extinction_time_mean[f] = np.mean(extinction_time)
        extinction_time_std[f] =  np.std(extinction_time)
        lam_net[f] = lam
        # N_net[f] = N
    os.chdir(dir_path)
    # plot_theory_well_mixed(lam, extinction_time_mean, extinction_time_std, N_net)
    plot_theory_well_mixed(lam_net, extinction_time_mean, extinction_time_std, N)

    # plot_theory_undirected_weak_hetro(lam, extinction_time_mean, extinction_time_std, N,eps_din_vec,eps_dout_vec)

def plot_weight(filename,directory_name):
    net_number = 0
    fig_extinction, ax_extinction = plt.subplots()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for f in range(len(filename)):
        os.chdir(dir_path+directory_name)
        os.chdir(filename[f])
        N, sims, it, k, x, lam, jump, Num_inf, number_of_networks,tau,eps_din,eps_dout = np.load('parameters.npy')
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
    directory_name ='/analysis/cluster/wellmixed/N350_sims_1000_and2000_jump2_and_5_net10/'
    filename =['wellmixed_N350_lam125_tau1_it100_jump5_quant5_sims1000_net10_eps0','wellmixed_N350_lam13_tau1_it100_jump5_quant5_sims1000_net10_eps0','wellmixed_N350_lam135_tau1_it100_jump5_quant5_sims1000_net10_eps0','wellmixed_N350_lam14_tau1_it100_jump2_quant5_sims2000_net10_eps0','wellmixed_N350_lam145_tau1_it100_jump2_quant5_sims2000_net10_eps0','wellmixed_N350_lam15_tau1_it100_jump2_quant5_sims2000_net10_eps0']
    # filename =['bimodal_N300_lam15_tau1_it100_jump2_quant5_sims1000_net10_eps02_mfix']
    # filename =['bimodal_N300_lam15_tau1_it100_jump2_quant5_sims1000_net10_eps005_mfix','bimodal_N300_lam15_tau1_it100_jump2_quant5_sims1000_net10_eps01_mfix','bimodal_N300_lam15_tau1_it100_jump2_quant5_sims1000_net10_eps02_mfix']
    filename = ['wellmixed_N350_lam14_tau1_it100_jump2_quant5_sims2000_net10_eps0']

    relaxation_time  = 15
    Alpha = 1.0
    # plot_MTE(filename,directory_name,relaxation_time)
    plot_weight(filename,directory_name)
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

def plot_MTE(filename,directory_name,relaxation_time):
    Alpha =1.0
    # extinction_time_mean, extinction_time_std,N_net = np.empty(len(filename)),np.empty(len(filename)),np.empty(len(filename))
    extinction_time_mean, extinction_time_std,lam_net = np.empty(len(filename)),np.empty(len(filename)),np.empty(len(filename))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    for f in range(len(filename)):
        os.chdir(dir_path+directory_name)
        os.chdir(filename[f])
        N, sims, it, k, x, lam, jump, Num_inf, number_of_networks,tau = np.load('parameters.npy')
        extinction_time =  np.empty(int(number_of_networks))
        for i in range(int(number_of_networks)):
            Death = np.load('Deaths_{}.npy'.format(i))
            extinction_time[i] = tau/np.mean((Death[relaxation_time:]))
        extinction_time_mean[f] = np.mean(extinction_time)
        extinction_time_std[f] =  np.std(extinction_time)
        # N_net[f] = N
        lam_net[f] = lam
    os.chdir(dir_path)
    # N_theory = np.linspace(np.min(N_net),np.max(N_net),100)
    lam_theory = np.linspace(np.min(lam_net),np.max(lam_net),100)
    # theory_well_mixed_mte = (1/Alpha)*np.sqrt(2*np.pi/N_theory)*(lam/(lam-1)**2)*np.exp(N_theory*(np.log(lam)+1/lam-1))
    theory_well_mixed_mte = (1/Alpha)*np.sqrt(2*np.pi/N)*(lam_theory/(lam_theory-1)**2)*np.exp(N*(np.log(lam_theory)+1/lam_theory-1))
    fig_extinction, ax_extinction = plt.subplots()
    # plt.scatter(N_net,np.log(extinction_time_mean))
    # plt.plot(N_theory,np.log(theory_well_mixed_mte))
    plt.errorbar(lam_net,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o')
    # plt.errorbar(N_net,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o')
    plt.plot(lam_theory,np.log(theory_well_mixed_mte))
    # plt.xlabel('N')
    plt.xlabel('R')
    plt.ylabel('ln(T)')
    # plt.title('N vs MTE N={}, R0={}'.format(N,lam))
    plt.title('R vs ln(T) N={}'.format(int(N)))
    # fig_extinction.savefig('extinction_v_N.png',dpi=200)
    fig_extinction.savefig('extinction_v_lam.png',dpi=200)
    plt.show()


def plot_weight(filename,directory_name):
    net_number = 0
    fig_extinction, ax_extinction = plt.subplots()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for f in range(len(filename)):
        os.chdir(dir_path+directory_name)
        os.chdir(filename[f])
        N, sims, it, k, x, lam, jump, Num_inf, number_of_networks,tau = np.load('parameters.npy')
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
    directory_name ='/analysis/wellmixed/'
    # filename =['wellmixed_N100_net20_it100_jump15_lam16_tau1','wellmixed_N200_net20_it100_jump15_lam16_tau1','wellmixed_N150_net20_it150_jump15_lam16_tau1','wellmixed_N250_net20_it150_jump15_lam16_tau1','wellmixed_N300_net20_it150_jump15_lam16_tau1']
    filename =['wellmixed_N400_lam12_tau1_it70_jump2_quant5_sims2000_net2','wellmixed_N400_lam125_tau1_it70_jump2_quant5_sims2000_net2','wellmixed_N400_net10_it150_jump10_lam13_tau1','wellmixed_N400_lam135_tau1_it70_jump2_quant5_sims2000_net2','wellmixed_N400_net10_it150_jump10_lam14_tau1','wellmixed_N400_lam15_tau1_it70_jump2_quant5_sims2000_net2']

    relaxation_time  = 15
    Alpha = 1.0
    plot_MTE(filename,directory_name,relaxation_time)
    # plot_weight(filename,directory_name)
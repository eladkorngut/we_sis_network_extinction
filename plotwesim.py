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
    extinction_time_mean, extinction_time_std,N_net = np.empty(len(filename)),np.empty(len(filename)),np.empty(len(filename))
    # extinction_time_mean, extinction_time_std,lam_net = np.empty(len(filename)),np.empty(len(filename)),np.empty(len(filename))

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
        N_net[f] = N
        # lam_net[f] = lam
    os.chdir(dir_path)
    theory_well_mixed_mte = (1/Alpha)*np.sqrt(2*np.pi/N_net)*(lam/(lam-1)**2)*np.exp(N_net*(np.log(lam)+1/lam-1))
    # theory_well_mixed_mte = (1/Alpha)*np.sqrt(2*np.pi/N)*(lam_net/(lam_net-1)**2)*np.exp(N*(np.log(lam_net)+1/lam_net-1))
    fig_extinction, ax_extinction = plt.subplots()
    # plt.scatter(N_net,np.log(extinction_time_mean))
    plt.plot(N_net,np.log(theory_well_mixed_mte))
    # plt.errorbar(lam_net,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o')
    plt.errorbar(N_net,np.log(extinction_time_mean),yerr=np.log(extinction_time_std)/np.log(extinction_time_mean),fmt='o')
    # plt.plot(lam_net,np.log(theory_well_mixed_mte))
    plt.xlabel('N')
    # plt.xlabel('lam_net')
    plt.ylabel('ln(T)')
    plt.title('N vs MTE N={}, R0={}'.format(N,lam))
    fig_extinction.savefig('extinction_v_N.png',dpi=200)
    # plt.title('R vs MTE N={}'.format(N))
    # fig_extinction.savefig('extinction_v_lam.png',dpi=200)
    plt.show()


def plot_weight(filename,directory_name):
    net_number = 5
    fig_extinction, ax_extinction = plt.subplots()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for f in range(len(filename)):
        os.chdir(dir_path+directory_name)
        os.chdir(filename[f])
        N, sims, it, k, x, lam, jump, Num_inf, number_of_networks,tau = np.load('parameters.npy')
        weight = np.load('weights_{}.npy'.format(net_number))
        # plt.scatter(range(len(weight)), weight)
        hist, edges = np.histogram(data, bins=10, weights=weight)
        # for i in range(int(number_of_networks)):
        #     weight = np.load('weights_{}_{}.npy'.format(i))
        #     plt.scatter(range(len(weight)), weight)
    os.chdir(dir_path)
    plt.xlabel('Bins')
    plt.ylabel('Weight')
    plt.title('Weight vs Bins N={}, R0={}'.format(N,lam))
    fig_extinction.savefig('Weight_v_Bins.png',dpi=200)
    plt.show()




if __name__ == '__main__':
    # Plot graphs of the we simulations
    directory_name ='/analysis/wellmixed/'
    filename =['wellmixed_N100_net20_it100_jump15_lam16_tau1','wellmixed_N200_net20_it100_jump15_lam16_tau1','wellmixed_N150_net20_it150_jump15_lam16_tau1','wellmixed_N250_net20_it150_jump15_lam16_tau1','wellmixed_N300_net20_it150_jump15_lam16_tau1']
    # filename =['wellmixed_N300_net20_it150_jump15_lam16_tau1']
    # filename =['wellmixed_N400_net10_it150_jump10_lam11_tau1','wellmixed_N400_net10_it150_jump10_lam13_tau1','wellmixed_N400_net10_it150_jump10_lam14_tau1','wellmixed_N400_net10_it150_jump10_lam15_tau1']

    relaxation_time  = 10
    Alpha = 1.0
    plot_MTE(filename,directory_name,relaxation_time)
    # plot_weight(filename,directory_name)
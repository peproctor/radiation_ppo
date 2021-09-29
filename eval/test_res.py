import joblib
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from statsmodels.stats.weightstats import DescrStatsW
from matplotlib.ticker import FormatStrFormatter
import math
import pickle

def make_env(random_ng, num_obs):
    import gym
    init_dims = {'bbox': [[0.0,0.0],[2700.0,0.0],[2700.0,2700.0],[0.0,2700.0]],
                'area_obs':[200.0,500.0], 'coord_noise':False,
                'obstruct':num_obs, 'seed' : random_ng}
    env_name = 'gym_rad_search:RadSearch-v0'
    env = gym.make(env_name,**init_dims)

    return env

def calc_stats(results,mc=None,plot=False,snr=None,control=None):
    stats = np.zeros((len(results[0]),len(results[0][0][1]),3))
    keys = results[0][0][1].keys()
    num_elem = 101
    d_count_dist = np.zeros((len(results[0]),2,num_elem))
    
    for jj, data in enumerate(results[0]):
        for ii, key in enumerate(keys):
            if 'Count' in key:
                stats[jj,ii,0:2] = data[1][key] if data[1][key].size > 0 else np.nan
            elif 'LocEstErr' in key:
                stats[jj,ii,0] = np.mean(data[1][key]) if data[1][key].size > 0 else np.nan
                stats[jj,ii,1] = np.var(data[1][key])/data[1][key].shape[0] if data[1][key].size > 0 else np.nan
            else:
                stats[jj,ii,0] = np.median(data[1][key]) if data[1][key].size > 0 else np.nan
                stats[jj,ii,1] = np.var(data[1][key])/data[1][key].shape[0] if data[1][key].size > 0 else np.nan
            stats[jj,ii,2] = data[1][key].shape[0]
            if key in 'dEpLen': #and isinstance(data[0],np.ndarray):
                uni,counts = np.unique(data[1][key],return_counts=True)
                sort_idx = np.argsort(counts)
                if len(sort_idx) > num_elem:
                    d_count_dist[jj,0,:] = uni[sort_idx][-num_elem:]
                    d_count_dist[jj,1,:] = counts[sort_idx][-num_elem:]
                else:
                    d_count_dist[jj,0,num_elem-len(sort_idx):] = uni[sort_idx][-num_elem:]
                    d_count_dist[jj,1,num_elem-len(sort_idx):] = counts[sort_idx][-num_elem:]
    #d_count_dist = np.delete(d_count_dist,np.all(d_count_dist[:,0,:] == 0,axis=1),axis=0)
    for ii, key in enumerate(keys):
        #if key in 'DoneCount':
        #    print('Sum '+ key +': ' +str(np.sum(stats[:,ii])/mc))
        if key in ['dIntDist','ndIntDist', 'dBkgDist','ndBkgDist','dEpRet','ndEpRet','ndEpLen','TotEpLen']:
            #print('Mean '+ key +': ' +str(np.round(np.nanmean(stats[:,ii]),decimals=2)))
            pass
        else:
            if 'LocEstErr' in key:
                tot_mean = np.mean(stats[:,ii,0])
                std_error = math.sqrt(np.nansum(stats[:,ii,1]/stats[:,ii,2]))
                print('Mean '+ key +': ' +str(np.round(tot_mean,decimals=2))+ ' +/- ' +str(np.round(std_error,3)))
            else:
                if np.nansum(stats[:,ii,1]) > 1:
                    d1 = DescrStatsW(stats[:,ii,0], weights=stats[:,ii,2])
                    lp_w, q1, weight_med, q3, hp_w = d1.quantile([0.025,0.25,0.5,0.75,0.975],return_pandas=False)
                    print('Weighted Median '+ key +': ' +str(np.round(weight_med,decimals=2))+ ' Weighted Percentiles (' +str(np.round(lp_w,3))+','+str(np.round(hp_w,3))+')')
                
                if plot:
                    #fig, ax = plt.subplots()
                    #box = ax.boxplot(stats[:,ii,0],usermedians=np.array([weight_med]),bootstrap=1000,sym='')
                    boxplot =  [{
                                    'whislo': lp_w,    # Bottom whisker position
                                    'q1'    : q1,    # First quartile (25th percentile)
                                    'med'   : weight_med,    # Median         (50th percentile)
                                    'q3'    : q3,    # Third quartile (75th percentile)
                                    'whishi': hp_w,    # Top whisker position
                                    'fliers': []        # Outliers
                                }]
                    #pickle.dump(boxplot, open('/Users/pproctor/Desktop/boxplots_pkl/'+control+'_'+snr+'_'+key+'_boxplot.pickle', 'wb'))
                    plt.close()
    return stats, d_count_dist

def common_elem(arr1,arr2,length):
    common_counts = {}
    diff_ls = []
    overlap_prev = [0,0]
    for num in range(10,61):
        if np.sum(np.any(num==arr1[:,0],axis=1))>10 and np.sum(np.any(num==arr2[:,0],axis=1)) > 10:
            common_counts[str(num)+'_arr1'] = [np.sum(np.any(num==arr1[:,0],axis=1)),np.sum(np.any(num==arr2[:,0],axis=1))]
            diff_ls.append(np.abs(common_counts[str(num)+'_arr1'][1] - common_counts[str(num)+'_arr1'][0]))
            overlap = np.where(np.any(num==arr1[:,0],axis=1) & np.any(num==arr2[:,0],axis=1))[0]
            common_counts[str(num)+'_arr1'].append(overlap)
            common_counts[str(num)+'_arr1'].append(len(overlap))
            if len(overlap) > overlap_prev[0]:
                overlap_prev = [len(overlap),num]
            #This is the number of times across the pop. that the ep length occurs:
            #EpLen: 24, 25 have even dist. for both controllers
            if (num == length):
                over_idx = np.where(np.any(num==arr1[:,0],axis=1) & np.any(num==arr2[:,0],axis=1),True,False)
                counts1 = arr1[over_idx,1,:][num==arr1[over_idx,0,:]]
                counts2 = arr2[over_idx,1,:][num==arr2[over_idx,0,:]]
                over_num = [len(overlap),num]
                #print(f'num of trials for nn: {(counts1 >= 5).sum()* 5 + counts1[(counts1 < 5)].sum()}')
                #print(f'num of trials for fic: {(counts2 >= 5).sum()* 5 + counts2[(counts2 < 5)].sum()}')
                print(f'num of trials for nn: {counts1.sum()}')
                print(f'num of trials for fic: {counts2.sum()}')
                print(f'shared eps for desired length: {over_num[0]}')
    print(f'Smallest diff: {np.min(diff_ls)} @ {np.argmin(diff_ls)}')
    print(f'{overlap_prev[0]} is largest num. of shared eps @ {overlap_prev[1]}')
    return common_counts, 0#over_num[1]

def plots(arr1,arr2,num_ep,arr3=None,size_=None,save_p=None,type_='loc',dist=False,ep=None):
    print(f'Tot samples for fic: {arr2[0][:,6,0].sum()}')
    print(f'Tot samples for NN: {arr1[0][:,6,0].sum()}')
    if type_ == 'loc':
        idx_1 = 1
        idx_2 = 3
        tit = 'Loc Est'
        ylab = 'RMSE [m]'
    else:
        idx_1 = 0
        idx_2 = 2
        tit = 'Int.'
        ylab = 'RMSE [cps]'
    fig, ax = plt.subplots()
    fig.set_size_inches((7,6))
    ax.plot(range(arr2[0].shape[2]),arr2[0][:,idx_2].mean(axis=0)/100,label='RID-FIM',c='orange')
    ax.plot(range(arr2[0].shape[2]),arr1[0][:,idx_2].mean(axis=0)/100,label='BPF-A2C',c='darkcyan')
    ax.plot(range(arr2[0].shape[2]),arr2[0][:,idx_1,:].mean(axis=0)/100,label='PCRB RID-FIM',linestyle='--',c='orange',alpha=0.7)
    ax.plot(range(arr2[0].shape[2]),np.nanmean(arr1[0][:,idx_1,:],axis=0)/100,label='PCRB BPF-A2C',linestyle='--',c='darkcyan',alpha=0.7)
    if isinstance(arr3,tuple):
        ax.plot(range(arr2[0].shape[2]),arr3[0][:,idx_2].mean(axis=0)/100,label='RAD-A2C',c='darkblue',alpha=0.7)
    #plt.title(tit+f" RMSE over {num_ep} episodes, FIC: {arr2[0][:,6,0].sum()}, NN:{arr1[0][:,6,0].sum()} runs")
    ax.set_xlabel('n')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_ylabel(ylab)
    ax.set_ylim(0,11)
    ax.set_xlim(0,arr2[0].shape[2]-1)
    #plt.ylabel('cps')
    plt.legend()
    if save_p and ((ep == 16) or (ep == 19) or (ep == 27)):
        if ep == 16:
            ax.xaxis.set_ticks(np.arange(0, arr2[0].shape[2], 2))
        else:
            ax.xaxis.set_ticks(np.arange(0, arr2[0].shape[2], 3))
        plt.savefig(save_p+f'fisher_bound_epi_{num_ep}_fic_{round(arr2[0][:,6,0].sum())}_nn_{round(arr1[0][:,6,0].sum())}.pdf',bbox_inches='tight')
        #plt.savefig(save_p+'.png',bbox_inches='tight')
        plt.close()
    else:
        plt.close()
    
    if type_ == 'loc' and 0:
        fig, ax = plt.subplots()
        fig.set_size_inches((7,6))
        ax.plot(range(1,arr2[0].shape[2]),np.log(arr2[0][:,4,1:].mean(axis=0)),label='RID-FIM')
        ax.plot(range(1,arr2[0].shape[2]),np.log(arr1[0][:,4,1:].mean(axis=0)),label='BPF-A2C')
        plt.legend(loc='upper left')
        ax.set_xlabel(r'n')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.set_ylabel(r'Log Fisher Score')
        ax.set_ylim(-8,16)
        ax.set_xlim(0,arr2[0].shape[2]-1)

        if save_p: #and ((ep == 16) or (ep == 19) or (ep == 27)):
            #plt.savefig(save_p+f'j_score_epi_{num_ep}_fic_{round(arr2[0][:,6,0].sum())}_nn_{round(arr1[0][:,6,0].sum())}_.png',bbox_inches='tight')
            plt.savefig(save_p+'.png',bbox_inches='tight')
            plt.close()
        else:
            plt.close()
        plt.figure()
        plt.plot(range(arr2[0].shape[2]),arr1[0][:,5,:].mean(axis=0)/100,label='RL-BPF')
        plt.plot(range(arr2[0].shape[2]),arr2[0][:,5,:].mean(axis=0)/100,label='RID-FIM',c='orange')
        plt.legend()
        plt.xlabel('n')
        plt.ylabel('RMSE [m]')
        #plt.title('Source-Detector Distance')

        if save_p and ((ep == 16) or (ep == 19) or (ep == 27)):
            #plt.savefig(save_p+f'det_dist_epi_{num_ep}_fic_{round(arr2[0][:,6,0].sum())}_nn_{round(arr1[0][:,6,0].sum())}_.png',bbox_inches='tight')
            plt.close()
        else:
            plt.close()

    if dist and ((ep == 16) or (ep == 19) or (ep == 27)):
        #snr = (arr2[1][:,0]/1e4 + arr2[1][:,1])/(arr2[1][:,1])
        #q1_snr  = np.percentile(snr, 25)
        #q3_snr  = np.percentile(snr, 75)
        q1_snr  = np.percentile(arr2[1][:,3], 25)
        q3_snr  = np.percentile(arr2[1][:,3], 75)
        q1_dist = np.percentile(arr2[1][:,2], 25)
        q3_dist = np.percentile(arr2[1][:,2], 75)
        bins_snr = 2*int((q3_snr-q1_snr)/(len(arr2[1][:,2])**(1/3))) + 2
        bins_dist = 2*int((q3_dist-q1_dist)/(len(arr2[1][:,2])**(1/3)))
        plt.figure()
        plt.hist(arr2[1][:,2],bins=bins_dist)
        plt.xlabel('Dist. [cm]')
        plt.ylabel('Counts')
        plt.title(f'Distribution of initial distance for {num_ep} episodes')
        if save_p:
            plt.savefig(save_p+'start_dist_2.png')
            plt.close()
        else:
            plt.close()
        
        plt.figure()
        #plt.hist(arr2[1][:,0]/1e4,bins=200)
        plt.hist(arr2[1][:,3],bins=bins_snr)
        plt.xlabel('SNR')
        plt.ylabel('Counts')
        plt.title(f'Distribution of SNR for {num_ep} episodes')
        if save_p:
            #plt.savefig(save_p+'snr_dist.png')
            plt.close()
        
    #plt.show()

def fish_calc(data,mc_ep_len,overlap,control,env_set,samps):
    results = np.zeros((len(overlap),7,mc_ep_len+1))
    pl_hold = np.zeros(mc_ep_len+1)
    env_dist = np.zeros((len(overlap),4))
    for ll, elem in enumerate(overlap):
        if np.any(data[0][elem][1]['TotEpLen'] == mc_ep_len):
            init_det_dist = np.linalg.norm(env_set['env_'+str(elem)][0]-env_set['env_'+str(elem)][1])
            env_dist[ll,:] = (env_set['env_'+str(elem)][2], env_set['env_'+str(elem)][3],init_det_dist, 
                              np.round(((env_set['env_'+str(elem)][2]/(init_det_dist**2))+env_set['env_'+str(elem)][3])/env_set['env_'+str(elem)][3],3))
            if samps < 0:
                len_ep = np.where(data[0][elem][1]['TotEpLen'] == mc_ep_len)[0]
            else:
                len_ep = np.where(data[0][elem][1]['TotEpLen'] == mc_ep_len)[0][:samps]
            fish_bound_arr = np.empty((len(len_ep),mc_ep_len+1,3))
            fish_bound = np.empty((mc_ep_len+1,6))
            loc_est = np.array([[data[0][elem][0][0][i]] for i in len_ep]).squeeze()
            FIM_bound = np.array([[data[0][elem][0][1][i]] for i in len_ep]).squeeze()
            J_score = np.array([[data[0][elem][0][2][i]] for i in len_ep]).squeeze()
            det_ls = np.array([[data[0][elem][0][3][i]] for i in len_ep]).squeeze()
            loc_est = loc_est[None,:,:] if len(loc_est.shape) < 3 else loc_est
            #print(f'J score shape: {J_score.shape}')
            J_score = J_score[None,:] if len(J_score.shape) < 2 else J_score
            if control == 'rid-fim':
                FIM_bound = FIM_bound[None,:,:] if len(FIM_bound.shape) < 4 else FIM_bound
                for ii,fim in enumerate(FIM_bound):
                    for zz in range(mc_ep_len+1):
                        fish_bound_arr[ii,zz,:] = np.diag(np.abs(inv(fim[zz])))#fisher_bound(FIM_bound[ii][zz])
                if len(det_ls.shape) < 3:
                    fish_bound[:,5]  = np.sqrt((np.linalg.norm(det_ls - env_set['env_'+str(elem)][0],axis=-1)**2))
                else:
                    fish_bound[:,5]  = np.sqrt((np.linalg.norm(det_ls - env_set['env_'+str(elem)][0],axis=-1)**2).mean(axis=0))
                fish_bound[1:,4] = J_score.mean(axis=0)
                fish_bound[:,3]  = np.sqrt((np.linalg.norm(loc_est[:,:,1:] - env_set['env_'+str(elem)][0],axis=2)**2).mean(axis=0))
                fish_bound[:,2]  = np.sqrt(np.square(loc_est[:,:,0] - (env_set['env_'+str(elem)][2]/1e4)).mean(axis=0))
                fish_bound[:,1]  = np.sqrt(np.sum(fish_bound_arr[:,:,1:],axis=2).mean(axis=0))#np.linalg.norm(fish_bound_arr[kk,:len_ep,1:],axis=1)
                fish_bound[:,0]  = np.sqrt(fish_bound_arr[:,:,0].mean(axis=0))  
            elif control == 'rad-a2c' :
                if len(det_ls.shape) < 3:
                    fish_bound[:,5]  = np.sqrt((np.linalg.norm(det_ls - env_set['env_'+str(elem)][0],axis=-1)**2))
                else:
                    fish_bound[:,5]  = np.sqrt((np.linalg.norm(det_ls - env_set['env_'+str(elem)][0],axis=-1)**2).mean(axis=0))
                fish_bound[:,4] = pl_hold
                fish_bound[:,3] = np.sqrt((np.linalg.norm(loc_est[:,1:,1:] - env_set['env_'+str(elem)][0],axis=2)**2).mean(axis=0))
                fish_bound[:,2] = np.sqrt(np.square(loc_est[:,:,0] - (env_set['env_'+str(elem)][2]/1e4)).mean(axis=0))
                fish_bound[:,1] = pl_hold#np.linalg.norm(fish_bound_arr[kk,:len_ep,1:],axis=1)
                fish_bound[:,0] = pl_hold
            else:
                FIM_bound = FIM_bound[None,:,:] if len(FIM_bound.shape) < 4 else FIM_bound
                for ii,fim in enumerate(FIM_bound):
                    for zz in range(mc_ep_len+1):
                        fish_bound_arr[ii,zz,:] = np.diag(np.abs(inv(fim[zz])))
                if len(det_ls.shape) < 3:
                    fish_bound[:,5]  = np.sqrt((np.linalg.norm(det_ls - env_set['env_'+str(elem)][0],axis=-1)**2))
                else:
                    fish_bound[:,5]  = np.sqrt((np.linalg.norm(det_ls - env_set['env_'+str(elem)][0],axis=-1)**2).mean(axis=0))
                fish_bound[1:,4] = J_score.mean(axis=0)
                fish_bound[:,3] = np.sqrt((np.linalg.norm(loc_est[:,:,1:] - env_set['env_'+str(elem)][0],axis=2)**2).mean(axis=0))
                fish_bound[:,2] = np.sqrt(np.square(loc_est[:,:,0] - (env_set['env_'+str(elem)][2]/1e4)).mean(axis=0))
                fish_bound[:,1] = np.sqrt(np.sum(fish_bound_arr[:,:,1:],axis=2).mean(axis=0))#np.linalg.norm(fish_bound_arr[kk,:len_ep,1:],axis=1)
                fish_bound[:,0] = np.sqrt(fish_bound_arr[:,:,0].mean(axis=0))  #pl_hold

            results[ll,0,:] = fish_bound[:,0]
            results[ll,1,:] = fish_bound[:,1]
            results[ll,2,:] = fish_bound[:,2]
            results[ll,3,:] = fish_bound[:,3]
            results[ll,4,1:]= fish_bound[1:,4]
            results[ll,5,:] = fish_bound[:,5]
            results[ll,6,:] = len(len_ep)
      
    return results, env_dist

def snr_hist(env_set):
    env_snr = []
    for key in env_set.keys():
        det = np.linalg.norm(env_set[key][0] - env_set[key][1])**2
        env_snr.append(((env_set[key][2]/det)+env_set[key][3])/env_set[key][3])
    bin_ = 15 - 0#74-15 #
    plt.figure()
    plt.hist(env_snr,bins=bin_)
    plt.show()

if __name__ == '__main__':
    """
    test_res.py calculates the statistics (weighted median, percentiles) from a set of runs generated by test_policy.py
    It also generates the PCRB plots for the BPF based methods. 
    If 1000 episodes and 100 MC runs have been performed, the loading can be slow
    """
    plt.rc('font',size=14)
    control_1 = 'rid-fim'
    control_2 = 'bpf-a2c'
    control_3 = 'rad-a2c'
    num = np.arange(10,57)

    #Maximum MC runs used per episode (keep the totals per algorithm the same), set to -1 to use all MC runs per calc.
    samples = 5 
    #Number of MC runs used in test policy.py
    mc = 100 
    #Number of episodes used in test policy.py
    n = 100
    snr = 'high'
    obs = 0
    seed = 0
    rng = np.random.default_rng(seed)
    env_fpath = 'test_envs/snr/test_env_dict_obs'
    env_path = env_fpath + str(obs) if snr is None else env_fpath + str(obs)+'_'+snr+'_v4'
    env = make_env(rng,obs)
    env_set = joblib.load(env_path)
    #snr_hist(env_set)
    res_bpf_a2c = joblib.load('results/raw/n_'+str(n)+'_mc'+str(mc)+'_'+control_2+'_freq_stats_'+snr+'_v4.pkl')
    res_fic = joblib.load('results/raw/n_'+str(n)+'_mc'+str(mc)+'_'+control_1+'_freq_stats_'+snr+'_v4.pkl')
    res_rad_a2c = joblib.load('results/raw/n_'+str(n)+'_mc'+str(mc)+'_'+control_3+'_freq_stats_'+snr+'_v4.pkl')

    #If the data has been processed already, load from processed
    load_rid_fim = False
    load_bpf_a2c  = False
    load_rad_a2c = False
    #Plot the PCRB and source prediction RMSE
    plot = False
    statistics = True
    sim_elem, key = common_elem(res_bpf_a2c,res_fic,num[28])
    num = np.arange(int(list(sim_elem.keys())[0][:2]),int(list(sim_elem.keys())[-5][:2])+1)

    if not load_rid_fim:
        print('Loading FIC results...')
        res_fic_full = joblib.load('results/raw/n_'+str(n)+'_mc'+str(mc)+'_'+control_1+'_full_dump_'+snr+'_v4.pkl')
        print('Loaded FIC results...')
        
        if statistics:
            calc_stats(res_fic_full,mc=100,plot=True,snr=snr,control=control_1)
            print('\n')
    else:
        print('Loading FIC results...')
        result_fic = joblib.load('results/processed/'+str(samples)+'_'+control_1+'_'+snr+'_processed_res.pkl')
        print('Loaded FIC results...')
        print('\n')

    if not load_bpf_a2c:
        print('Loading BPF-A2C results...')
        res_bpf_a2c_full = joblib.load('results/raw/n_'+str(n)+'_mc'+str(mc)+'_'+control_2+'_full_dump_'+snr+'_v4.pkl')
        print('Loaded BPF-A2C results...')
        if statistics:
            calc_stats(res_bpf_a2c_full,mc=100,plot=True,snr=snr,control=control_2)
            print('\n')
    else:
        print('Loading BPF-A2C results...')
        result_bpf_a2c = joblib.load('results/processed/'+str(samples)+'_'+control_2+'_'+snr+'_processed_res.pkl')
        print('Loaded BPF-A2C results...')
        print('\n')

    if not load_rad_a2c:
        print('Loading RAD-A2C results...')
        res_rad_a2c_full = joblib.load('results/raw/n_'+str(n)+'_mc'+str(mc)+'_'+control_3+'_full_dump_'+snr+'_v4.pkl')
        print('Loaded RAD-A2C results...')
        
        if statistics:
            calc_stats(res_rad_a2c_full,mc=100,plot=True,snr=snr,control=control_3)
            print('\n')

    if not load_rid_fim or not load_bpf_a2c:
        result_fic = []
        result_bpf_a2c = []
        for key in num:
            if not load_rid_fim:
                result_fic.append(fish_calc(res_fic_full,key,sim_elem[str(key)+'_arr1'][2],control_1,env_set,samples))
            if not load_bpf_a2c:
                result_bpf_a2c.append(fish_calc(res_bpf_a2c_full,key,sim_elem[str(key)+'_arr1'][2],control_2,env_set,samples))
        
        if not load_rid_fim:
            print('Saving FIC processed results...')
            joblib.dump(result_fic,'results/processed/'+str(samples)+'_'+control_1+'_'+snr+'_processed_res.pkl')
        if not load_bpf_a2c:
            print('Saving NN processed results...')
            joblib.dump(result_bpf_a2c,'results/processed/'+str(samples)+'_'+control_2+'_'+snr+'_processed_res.pkl')

    if plot:
        for jj,key in enumerate(num):
            if result_bpf_a2c[jj][0].shape[0] >= (mc*0.15):
                plots(result_bpf_a2c[jj],result_fic[jj],0,arr3=None,dist=False,ep=key,
                    size_=key+1,type_='loc',save_p='results/figs/'+snr+'_'+str(samples)+'_n_999_l_'+str(key)+'_')
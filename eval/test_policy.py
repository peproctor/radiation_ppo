import time
import joblib
import os
import torch
import numpy as np
from numpy.linalg import inv
import core
import matplotlib.pyplot as plt
import math
import visilibity as vis
from scipy.stats import norm
from gym.utils.seeding import _int_list_from_bigint, hash_seed
from functools import partial
from multiprocessing import Pool
from statsmodels.stats.weightstats import DescrStatsW

DET_STEP = 100
DET_STEP_FRAC = 71.0
ACTION   = np.array([[-DET_STEP,0],
                    [-DET_STEP_FRAC,DET_STEP_FRAC],
                    [0,DET_STEP],
                    [DET_STEP_FRAC,DET_STEP_FRAC],
                    [DET_STEP,0],
                    [DET_STEP_FRAC,-DET_STEP_FRAC],
                    [0,-DET_STEP],
                    [-DET_STEP_FRAC,-DET_STEP_FRAC]])
EPSILON = 0.0000001


def make_env(random_ng, num_obs):
    """Create radiation source search environment"""
    import gym
    init_dims = {'bbox': [[0.0,0.0],[2700.0,0.0],[2700.0,2700.0],[0.0,2700.0]],
                'area_obs':[200.0,500.0], 'coord_noise':False,
                'obstruct':num_obs, 'seed' : random_ng}
    env_name = 'gym_rad_search:RadSearch-v0'
    env = gym.make(env_name,**init_dims)

    return env


def select_model(fpath, ac_kwargs, bp_kwargs, grad_kwargs, model='rl',
                        bkg=None, FIM_step=None,search_area_max=None, scale=None,env=None):
    """ 
    Sets which algorithm will be used for action selection (get_action fcn.).
    See "gs" conditional branch for further comments
    Choices are: 
    gs: gradient search 
    bpf-a2c: bootstrap particle filter with actor critic
    rad-a2c: particle filter GRU with actor critic
    rid-fim: bootstrap particle filter with renyi information divergence-fisher information matrix controller
    """
    from gym.spaces import Box, Discrete
    fname = os.path.join(fpath, 'pyt_save', 'model'+'.pt')
    #print('\n\nLoading from %s.\n\n'%fname)
     
    obs_space = Box(0, np.inf, shape=(11,), dtype=np.float32)
    act_space = Discrete(8)
    
    bp_kwargs['bkg'] = bkg
    bp_kwargs['scale'] = scale
    if model =='gs':
        #Instantiate model
        grad_act = core.GradSearch(**grad_kwargs)
        fim = np.zeros((3,3))
        x_est_glob = np.zeros((3,))
        def get_action(x,est=False,FIM=False,act=False,hidden=None,done=False,post=0,step=None,init_est=False):
            """
            Args:
            If est is true:
                x (list): contains the raw (unprocessed) radiation measurement and detector coordinates
            If init_est is true:
                x (list): in order, contains the location prediction, unnormalized detector coordinates,
                          background rate, scaling matrix, prior probabilities for BPF estimates,
                          standardized observation
            If act is true:
                x (list): in order, contains the location prediction, raw (unprocessed) radiation measurement and detector coordinates,
                          processed radiation measurement and detector coordinates, background rate, scaling matrix
            
            """

            nonlocal x_est_glob
            if est:
                #Location prediction if applicable
                return x_est_glob.copy()
            elif FIM or FIM is 0:
                #Calculate Fisher information if applicable
                return fim
            elif init_est:
                #Initial location prediction if applicable
                return fim,x_est_glob.copy()
            elif act:
                #Action selection
                return grad_act.step(x[1]), x_est_glob, 0
            else:
                #Reset model if when doing multiple runs in same environment
                x_est_glob = np.zeros((3,))
    elif model == 'bpf-a2c':
        ac = core.RNNModelActorCritic(obs_space,act_space,**ac_kwargs)
        ac.load_state_dict(torch.load(fname))
        ac.model = core.ParticleFilter(**bp_kwargs)
        ac.eval()
        fim = np.zeros((3,3))
        def get_action(x,est=False,FIM=False,act=False,hidden=None,done=False,post=0,step=None,init_est=False):
            if est:
                return ac.model.track(x)
            elif FIM or FIM is 0:
                pred = ac.model.xp_prev[:,FIM,:].copy()
                pred[:,0] = pred[:,0]*1e4
                denom = np.sum(np.square(x[1][1:] - pred[:,1:]),axis=1)
                grad_xy = (2*(x[1][1:] - pred[:,1:]))*(pred[:,0] / np.square(denom))[:,None]
                grad_I = 1 / denom
                grad = np.hstack((grad_I[:,None], grad_xy))
                J = np.einsum('ij,ikl->ijk',grad,grad[:,:,None])* np.tile((1/(pred[:,0]/denom + x[2]))[:,None,None],(3,3))
                return (((J@x[3])*(ac.model.wp_prev[:,FIM,None,:])).sum(axis=0)).squeeze()
            elif init_est:
                pred = ac.model.xp_init.copy()
                pred[:,0] = pred[:,0]*1e4
                denom = np.sum(np.square(x[1] - pred[:,1:]),axis=1)
                grad_xy = (2*(x[1][1:] - pred[:,1:]))*(pred[:,0] / np.square(denom))[:,None]
                grad_I = 1 / denom
                grad = np.hstack((grad_I[:,None], grad_xy))
                J = np.einsum('ij,ikl->ijk',grad,grad[:,:,None])* np.tile((1/((pred[:,0]/denom) + x[2]))[:,None,None],(3,3))
                return (J@(x[3]@x[4])).mean(axis=0).squeeze(),(ac.model.xp_init * ac.model.wp_init).sum(axis=0)
            elif act:
                if hidden is None:
                    hidden = ac.reset_hidden()
                
                with torch.no_grad():
                    x_obs = torch.FloatTensor(np.append(x[2],x[0][1:]/search_area_max)[None,None,:])
                    action, hidden, _ = ac.pi._distribution(x_obs,hidden=hidden[1])
                    act = action.sample().item()
                    
                    sim_det = sim_step(act,x[1][1:])
                    pred = ac.model.xp_prev[:,step,:].copy()
                    pred[:,0] = pred[:,0]*1e4
                    denom = np.sum(np.square(sim_det - pred[:,1:]),axis=1)
                    grad_xy = (2*(sim_det - pred[:,1:]))*(pred[:,0] / np.square(denom))[:,None]
                    grad_I = 1 / denom
                    grad = np.hstack((grad_I[:,None], grad_xy))
                    J = np.einsum('ij,ikl->ijk',grad,grad[:,:,None])* np.tile((1/(pred[:,0]/denom + x[3]))[:,None,None],(3,3))
                    J_tot = np.trace(((J@x[4])*(ac.model.wp_prev[:,step,None,:])).sum(axis=0).squeeze())
                    return act, J_tot,(None,hidden)
            else:
                ac.model = core.ParticleFilter(**bp_kwargs)

    elif model == 'rad-a2c':
        ac = core.RNNModelActorCritic(obs_space,act_space,**ac_kwargs)
        ac.load_state_dict(torch.load(fname))
        ac.eval()
        fim = np.zeros((3,3))
        x_est_glob = np.zeros((3,))
        def get_action(x,est=False,FIM=False,act=False,hidden=None,done=False,post=0,step=None,init_est=False):
            nonlocal x_est_glob
            if est:
                return x_est_glob.copy()
            elif FIM or FIM is 0:
                return fim
            elif init_est:
                hidden = ac.reset_hidden()[0]
                with torch.no_grad():
                    x_est_glob[1:],_ = ac.model( torch.as_tensor(x[5][:3], dtype=torch.float32).unsqueeze(0),hidden)
                return fim,x_est_glob.copy()
            elif act:
                if hidden is None:
                    hidden = ac.reset_hidden()
                with torch.no_grad():
                    action, _, _, hidden, x_est_glob[1:] = ac.act(x[2],hidden=hidden)
                    return action, x_est_glob, hidden
            else:
                x_est_glob = np.zeros((3,))
    elif model == 'rid-fim':
        ac = core.FIC(**bp_kwargs)
        ac.FIM_step = FIM_step
        def get_action(x,est=False,FIM=False,act=False,hidden=None,done=False,post=0,step=None,init_est=False):
            if done:
                print('Tracking marg!')
                ac.bpf.marg_mp(5000)
                ac.bpf.plot_marg(x)
            elif est:
                return ac.bpf.track(x)
            elif init_est:
                pred = ac.bpf.xp_init.copy()
                pred[:,0] = pred[:,0]*1e4
                denom = np.sum(np.square(x[1][1:] - pred[:,1:]),axis=1)
                grad_xy = (2*(x[1][1:] - pred[:,1:]))*(pred[:,0] / np.square(denom))[:,None]
                grad_I = 1 / denom
                grad = np.hstack((grad_I[:,None], grad_xy))
                J = np.einsum('ij,ikl->ijk',grad,grad[:,:,None])* np.tile((1/((pred[:,0]/denom) + x[2]))[:,None,None],(ac.s_size,ac.s_size))
                return (((J@(x[3]@x[4])).mean(axis=0)).squeeze(),(ac.bpf.xp_init * ac.bpf.wp_init).sum(axis=0))
            elif FIM or FIM is 0:
                pred = ac.bpf.xp_prev[:,FIM,:].copy()
                pred[:,0] = pred[:,0]*1e4
                denom = np.sum(np.square(x[1][1:] - pred[:,1:]),axis=1)
                grad_xy = (2*(x[1][1:] - pred[:,1:]))*(pred[:,0] / np.square(denom))[:,None]
                grad_I = 1 / denom
                grad = np.hstack((grad_I[:,None], grad_xy))
                J = np.einsum('ij,ikl->ijk',grad,grad[:,:,None])* np.tile((1/(pred[:,0]/denom + x[2]))[:,None,None],(ac.s_size,ac.s_size))
                return (((J@x[3])*(ac.bpf.wp_prev[:,FIM,None,:])).sum(axis=0)).squeeze()
            elif act:
                ret = ac.optim_action(np.append(x[-1],x[1][1:]),x[0],step=step)
                return ret[0],ret[1], None

            elif post:
                probs = np.zeros((ac.bpf.state_dim,ac.bpf.state_dim))
                covar = np.diag(np.square(1/post[1]))

                for jj in range(ac.bpf.nPart):
                    mu = (ac.bpf.xp_prev[jj,post[0],None] - ac.bpf.xp_prev[:,post[0]-1]).squeeze()
                    w_T = ac.bpf.wp_prev[:,post[0]-1].T
                    p_x = norm.pdf(mu/post[1])/post[1]
                    phi = (w_T @ p_x).squeeze()
                    grad = 1*(w_T @ (p_x * (mu @ covar))).squeeze()
                    grad_op = np.outer(grad,grad)
                    probs += ((grad_op)/(phi[:,None]**2)) * ac.bpf.wp_prev[jj,post[0]]
                
                return probs
            else:
                ac.bpf = core.ParticleFilter(**bp_kwargs)
    else:
        raise ValueError('Invalid model type!')
    return get_action

def set_vis_coord(point, coords):
    point.set_x(coords[0])
    point.set_y(coords[1])
    return point

def sim_step(act,det):
    return det + ACTION[act]

def refresh_env(env_dict,env,n,num_obs=0):
    """
    Load saved test environment parameters from dictionary
    into the current instantiation of environment
    """
    key = 'env_'+str(n)
    env.src_coords    = env_dict[key][0]
    env.det_coords    = env_dict[key][1].copy()
    env.intensity     = env_dict[key][2]
    env.bkg_intensity = env_dict[key][3]
    env.source        = set_vis_coord(env.source,env.src_coords)
    env.detector      = set_vis_coord(env.detector,env.det_coords)
    
    if num_obs > 0:
        env.obs_coord = env_dict[key][4]
        env.num_obs = len(env_dict[key][4])
        env.poly = []
        env.line_segs = []
        for obs in env.obs_coord:
            geom = [vis.Point(float(obs[0][jj][0]),float(obs[0][jj][1])) for jj in range(len(obs[0]))]
            poly = vis.Polygon(geom)
            env.poly.append(poly)
            env.line_segs.append([vis.Line_Segment(geom[0],geom[1]),vis.Line_Segment(geom[0],geom[3]),
            vis.Line_Segment(geom[2],geom[1]),vis.Line_Segment(geom[2],geom[3])]) 
        
        env.env_ls = [solid for solid in env.poly]
        env.env_ls.insert(0,env.walls)
        env.world = vis.Environment(env.env_ls)
        # Check if the environment is valid
        assert env.world.is_valid(EPSILON), "Environment is not valid"
        env.vis_graph = vis.Visibility_Graph(env.world, EPSILON)

    o, _, _, _        = env.step(-1)
    env.det_sto       = [env_dict[key][1].copy()]
    env.src_sto       = [env_dict[key][0].copy()]
    env.meas_sto      = [o[0].copy()]
    env.prev_det_dist = env.world.shortest_path(env.source,env.detector,env.vis_graph,EPSILON).length()
    env.iter_count    = 1
    return o, env

def calc_stats(results,mc=None,plot=False,snr=None,control=None,obs=None):
    """Calculate results from the evaluation"""
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

    for ii, key in enumerate(keys):
        if key in ['dIntDist','ndIntDist', 'dBkgDist','ndBkgDist','dEpRet','ndEpRet','ndEpLen','TotEpLen']:
            pass
        else:
            if 'LocEstErr' in key:
                tot_mean = np.mean(stats[:,ii,0])
                std_error = math.sqrt(np.nansum(stats[:,ii,1]/stats[:,ii,2]))
                #print('Mean '+ key +': ' +str(np.round(tot_mean,decimals=2))+ ' +/- ' +str(np.round(std_error,3)))
            else:
                if np.nansum(stats[:,ii,0]) > 1:
                    d1 = DescrStatsW(stats[:,ii,0], weights=stats[:,ii,2])
                    lp_w, weight_med, hp_w = d1.quantile([0.025,0.5,0.975],return_pandas=False)
                    q1, q3 = d1.quantile([0.25,0.75],return_pandas=False)
                    print('Weighted Median '+ key +': ' +str(np.round(weight_med,decimals=2))+ ' Weighted Percentiles (' +str(np.round(lp_w,3))+','+str(np.round(hp_w,3))+')')

    return stats, d_count_dist

def run_policy(env, env_set, render=True, save_gif=False, save_path=None, 
               MC=1, control='fic', fish_analysis=False, ac_kwargs=None, 
               bp_kwargs=None, grad_kwargs=None,tot_ep=1, n=0):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."
    
    #Setup intial data structures
    ep_ret_ls = []
    loc_est_ls = []
    FIM_bound = [] 
    render_num = []
    mc_stats = {}
    done_count = 0
    repl = 0
    mc = 0
    seq_sto = {}
    done_dist_int, done_dist_bkg, not_done_dist_int, not_done_dist_bkg = np.array([]), np.array([]), np.array([]), np.array([])
    tot_ep_len, d_ep_ret, nd_ep_ret, d_ep_len, nd_ep_len = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    FIM_bound = [[] for _ in range(MC)]
    loc_est_ls = [[] for _ in range(MC)]
    J_score_ls = [[] for _ in range(MC)]
    det_ls = [[] for _ in range(MC)]
    loc_est_err = np.array([])

    #Set A2C hidden state to initial condition
    hidden = None

    #Scaling and prior probabilities for PCRB calculation 
    scale_mat = np.diag(np.array([1e10,1,1]))
    uni_probs = np.diag(np.array([(1/(1e3-1e2)), 
                    (1/(25e2+0)),
                    (1/(25e2+0))]))

    #Variances for PCRB calcs.
    sigma_mat = np.array([bp_kwargs['noise_params'][1],
                          bp_kwargs['noise_params'][0],
                          bp_kwargs['noise_params'][0]])
    pro_covar = inv(np.diag(np.square(sigma_mat))) 

    #Reset environment and then replace the original parameter with test set parameters
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    o, env = refresh_env(env_set,env,n,num_obs=len(env.obs_coord))

    #Instantiate and update running standardization module
    stat_buff = core.StatBuff()
    stat_buff.update(o[0])

    max_ep_len= env._max_episode_steps
   
    #Set the algorithm that will be used for action selection
    get_action = select_model(save_path, ac_kwargs, bp_kwargs, grad_kwargs, model=control, 
                                     bkg=env.bkg_intensity, FIM_step=env.FIM_step, search_area_max=env.search_area[2][1],
                                     scale=scale_mat,env=env)



    #Make initial location prediction if applicable
    x_est = get_action(np.append((env.meas_sto[ep_len]),env.det_sto[ep_len]),est=True)
    obs_std = o
    obs_std[0] = np.clip((o[0]-stat_buff.mu)/stat_buff.sig_obs,-8,8) 
    
    #Get initial FIM calculation if applicable
    est_init_bnd, x_est_init = get_action([x_est,env.det_sto[ep_len],env.bkg_intensity,scale_mat,uni_probs[None,:],obs_std],init_est=True) 
    loc_est_ls[mc].append(x_est_init)
    FIM_bound[mc].append(est_init_bnd)

    while mc < MC: #Perform Monte Carlo runs
        loc_est_ls[mc].append(x_est)

        #Get action, fisher score, and hidden state when applicable
        action, score, hidden = get_action([x_est,np.append(env.meas_sto[ep_len],env.det_sto[ep_len]),obs_std,env.bkg_intensity,scale_mat],
                                            hidden=hidden,act=True,step=ep_len)

        #Take action in environment and get new observation
        o, r, d, _ = env.step(action)

        #Update running statistics
        stat_buff.update(o[0])
        obs_std = o
        obs_std[0] = np.clip((o[0]-stat_buff.mu)/stat_buff.sig_obs,-8,8) 

        ep_ret += r
        ep_len += 1
        J_score_ls[mc].append(score)
        ep_ret_ls.append(ep_ret)
        
        #Next location prediction
        x_est = get_action(np.append((env.meas_sto[ep_len]),env.det_sto[ep_len]),est=True) 
        
        if fish_analysis:
            #Calculate PCRB if an algorithm is using the bootstrap particle filter
            R_t = get_action([x_est,env.det_sto[ep_len],env.bkg_intensity,scale_mat],FIM=ep_len)
            rec_bpf = pro_covar + R_t - np.square(pro_covar) @ inv(FIM_bound[mc][ep_len-1]  + pro_covar)
            FIM_bound[mc].append(rec_bpf)
        
        if d or (ep_len == max_ep_len):
            if control == 'rad-a2c':
                loc_est_ls[mc].append(x_est)
                loc_est_ls[mc] = np.delete(loc_est_ls[mc],1,axis=0) * env.search_area[2][1]
                loc_est_err =np.append(loc_est_err, math.sqrt(np.sum(np.square(loc_est_ls[mc][:,1:] - env.src_coords),axis=1).mean()))    
            else:
                loc_est_err = np.append(loc_est_err,math.sqrt(np.sum(np.square(np.array(loc_est_ls[mc])[:,1:] - env.src_coords),axis=1).mean()))
            det_ls[mc].append(np.array(env.det_sto))
            if mc < 1:
                if d:
                    done_dist_int = np.append(done_dist_int,env.intensity)
                    done_dist_bkg = np.append(done_dist_bkg,env.bkg_intensity)
                else:
                    not_done_dist_int = np.append(not_done_dist_int,env.intensity)
                    not_done_dist_bkg = np.append(not_done_dist_bkg,env.bkg_intensity)
            tot_ep_len = np.append(tot_ep_len,ep_len)
            if d:
                done_count += 1
                d_ep_len = np.append(d_ep_len,ep_len)
                d_ep_ret = np.append(d_ep_ret,ep_ret)
            else:
                nd_ep_len = np.append(nd_ep_len,ep_len)
                nd_ep_ret = np.append(nd_ep_ret,ep_ret)

            if render and n==(tot_ep-1) and repl < 1: 
                #Save trajectory for future rendering
                seq_sto['Ep'+str(mc)+'_rew'] = ep_ret_ls
                seq_sto['Ep'+str(mc)+'_meas'] = env.meas_sto
                seq_sto['Ep'+str(mc)+'_det'] = env.det_sto
                seq_sto['Ep'+str(mc)+'_params'] = [env.intensity,env.bkg_intensity,env.src_coords]
                seq_sto['Ep'+str(mc)+'_obs'] = env.obs_coord
                seq_sto['Ep'+str(mc)+'_loc'] = loc_est_ls
                render_num.append(mc) 
                repl += 1

            mc += 1
            
            #Reset environment without performing an env.reset
            env.epoch_end = False
            env.done = False; env.oob = False
            env.iter_count = 0
            env.oob_count = 0
            r, d, ep_ret, ep_len = 0, False, 0, 0
            o, env = refresh_env(env_set,env,n,num_obs=len(env.obs_coord))

            #Reset running statistics, hidden state initial condition and 
            ep_ret_ls= []
            stat_buff.reset()
            stat_buff.update(o[0])
            obs_std = o
            obs_std[0] = np.clip((o[0]-stat_buff.mu)/stat_buff.sig_obs,-8,8)
            hidden = None

            #Reset model in action selection fcn.
            get_action(0)
            
            #Get initial location prediction
            x_est = get_action(np.append((env.meas_sto[ep_len]),env.det_sto[ep_len]),est=True)
            if fish_analysis and mc < MC:
                est_init_bnd, x_est_init = get_action([x_est,env.det_sto[ep_len],env.bkg_intensity,scale_mat,uni_probs[None,:],obs_std],init_est=True) 
                est_init_bnd = (est_init_bnd) 
                loc_est_ls[mc].append(x_est_init)
                FIM_bound[mc].append(est_init_bnd)
            

    if render and n==(tot_ep-1):
        for i in render_num:
            env.render(data=seq_sto['Ep'+str(i)+'_det'],
                       meas=seq_sto['Ep'+str(i)+'_meas'],
                       ep_rew=seq_sto['Ep'+str(i)+'_rew'],
                       params=seq_sto['Ep'+str(i)+'_params'],
                       obs=seq_sto['Ep'+str(i)+'_obs'],
                       loc_est=seq_sto['Ep'+str(i)+'_loc'],
                       save_gif=save_gif,
                       just_env=False,
                       path=save_path,epoch_count=i)

        time.sleep(1e-3)
    
    mc_stats['dEpLen'] = d_ep_len
    mc_stats['ndEpLen'] = nd_ep_len
    mc_stats['dEpRet'] = d_ep_ret
    mc_stats['ndEpRet'] = nd_ep_ret
    mc_stats['dIntDist'] = done_dist_int
    mc_stats['ndIntDist'] = not_done_dist_int
    mc_stats['dBkgDist'] = done_dist_bkg
    mc_stats['ndBkgDist'] = not_done_dist_bkg
    mc_stats['DoneCount'] = np.array([done_count])
    mc_stats['TotEpLen'] = tot_ep_len
    mc_stats['LocEstErr'] = loc_est_err
    results = [loc_est_ls, FIM_bound, J_score_ls, det_ls]
    print(f'Finished run {n}!, done count: {done_count}')
    return (results,mc_stats)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str,default='../models/train/bpf/loc24_hid24_pol32_val32_alpha01_tkl07_val01_lam09_npart40_lr3e-4_proc10_obs-1_iter40_blr5e-3_2_tanh_5_ep3000_steps4800_s1',
    help='Specify model directory, Ex: ../models/train/bpf/model_dir')
    parser.add_argument('--episodes', '-n', type=int, default=1000,help='Number of episodes to test on, option: [1-1000]') 
    parser.add_argument('--render', '-r',type=bool, default=False,help='Produce gif of agent in environment')
    parser.add_argument('--save_gif', type=bool,default=False, help='Save gif of the agent in model folder, render must be true')
    parser.add_argument('--control', type=str, default='rad-a2c',help='Control algorithm, options: [rad-a2c,bpf-a2c,gs,rid-fim]')
    parser.add_argument('--snr', type=str, default='high',help='SNR of environment, options: [low,med,high]')
    parser.add_argument('--num_obs', type=int, default=1,help='Number of obstructions in environment, options:[1,3,5,7]')
    parser.add_argument('--mc_runs', type=int, default=1,help='Number of Monte Carlo runs per episode')
    parser.add_argument('--num_cpu', '-ncpu', type=int, default=1,help='Number of cpus to run episodes across')
    parser.add_argument('--fisher',type=bool, default=False,help='Calculate the posterior Cramer-Rao Bound for BPF based methods')
    parser.add_argument('--save_results', type=bool, default=False, help='Save list of results across episodes and runs')
    args = parser.parse_args()
    
    
    plt.rc('font',size=14)
    seed = 9389090
    #Path for the test environments
    env_fpath = 'test_envs/snr/test_env_dict_obs'
    robust_seed = _int_list_from_bigint(hash_seed(seed))[0]
    rng = np.random.default_rng(robust_seed)
    params = np.arange(0,args.episodes,1)

    #Load set of test envs 
    env = make_env(rng,args.num_obs)
    env_path = env_fpath + str(args.num_obs) if args.snr is None else env_fpath + str(args.num_obs)+'_'+args.snr+'_v4'
    env_set = joblib.load(env_path)

    #Model parameters, must match the model being loaded
    ac_kwargs = {'batch_s': 1, 'hidden': [24], 'hidden_sizes_pol': [32], 'hidden_sizes_rec': [24], 
                 'hidden_sizes_val': [32], 'net_type': 'rnn', 'pad_dim': 2, 'seed': robust_seed}

    #Bootstrap particle filter parameters for RID-FIM controller and BPF-A2C
    bp_kwargs = {'nParticles':int(6e3), 'noise_params':[15.,1.,1],'thresh':1,'s_size':3,
                'rng':rng, 'L': 1,'k':0.0, 'alpha':0.6, 'fim_thresh':0.36,'interval':[75,75]}
    
    #Gradient search parameters
    grad_kwargs = {'q':0.0042,'env':env}

    #Create partial func. for use with multiprocessing
    func = partial(run_policy, env, env_set, args.render,args.save_gif, 
                   args.fpath, args.mc_runs, args.control,args.fisher,
                   ac_kwargs,bp_kwargs,grad_kwargs, args.episodes)
    
    mc_results = []
    print(f'Number of cpus available: {os.cpu_count()}')
    print('Starting pool')
    p = Pool(processes=args.num_cpu)
    mc_results.append(p.map(func,params)) 
    stats, len_freq = calc_stats(mc_results,mc=args.mc_runs,plot=False,snr=args.snr,control=args.control,obs=args.num_obs)
    print('Saving results..')
    
    if args.save_results:
        joblib.dump(stats,'results/raw/n_999_bpf_mc'+str(args.mc_runs)+'_'+args.control+'_'+'eplen_fim_r_div_scale_'+args.snr+'_v4.pkl')
        joblib.dump(len_freq,'results/raw/n_999_bpf_mc'+str(args.mc_runs)+'_'+args.control+'_'+'eplen_fim_r_div_freq_len_scale_inv_'+args.snr+'_v4.pkl')
        joblib.dump(mc_results,'results/raw/n_999_bpf_mc'+str(args.mc_runs)+'_'+args.control+'_'+'eplen_fim_r_div_full_dump_scale_inv_'+args.snr+'_v4.pkl')
      
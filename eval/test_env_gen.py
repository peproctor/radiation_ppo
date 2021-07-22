import gym
import numpy as np
import joblib
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import visilibity as vis
from gym.utils.seeding import _int_list_from_bigint, hash_seed

EPSILON = 0.0000001

def create_envs(num_envs, init_dims, env_name, save_path):
    env_dict = {}
    for ii in range(num_envs):
        env = gym.make(env_name,**init_dims)
        env.reset()
        if init_dims['obstruct'] > 0 or init_dims['obstruct'] == -1:
            env_dict['env_'+str(ii)] =  (env.src_coords,env.det_coords,env.intensity,env.bkg_intensity, env.obs_coord)
        else:
            env_dict['env_'+str(ii)] =  (env.src_coords,env.det_coords,env.intensity,env.bkg_intensity)
            print(f'Source coord: {env.src_coords}, Det coord: {env.det_coords}, Intensity: {env.intensity},{env.bkg_intensity}')

    joblib.dump(env_dict, osp.join(save_path, 'test_env_dict_obs'+str(init_dims['obstruct'])))

def create_envs_snr(num_envs, init_dims, env_name, save_path,split=4, snr='low'):
    env_dict = {}
    ii = 0
    rng = init_dims['seed'] #lowest snr = 1.0, highest snr = 2.0
    snr_range = {'none':[0,0],'low':[1.0,1.2], 'med':[1.2,1.6], 'high':[1.6,2.0]} 
    div = np.round((snr_range[snr][1] - snr_range[snr][0])/(split),2)
    num_envs_split = round(num_envs / (split))
    counts = np.zeros(split)
    while ii < num_envs:
        env = gym.make(env_name,**init_dims)
        env.reset()
        det = np.linalg.norm(env.src_coords - env.det_coords)
        meas = env.intensity/(det**2) + env.bkg_intensity 
        if snr == 'none':
            if init_dims['obstruct'] > 0 or init_dims['obstruct'] == -1:         
                env_dict['env_'+str(ii)] =  (env.src_coords,env.det_coords,env.intensity,env.bkg_intensity, env.obs_coord)
                ii+=1
            else:
                env_dict['env_'+str(ii)] =  (env.src_coords,env.det_coords,env.intensity,env.bkg_intensity)
                ii += 1
        else:
            snr_exp = meas / env.bkg_intensity
            if snr_range[snr][0] < snr_exp <= snr_range[snr][1]:
                if snr == 'med' or snr == 'low' or snr == 'high':
                    counts, inc_flag = classify_snr(np.round(snr_exp,3), div,counts, num_envs_split,lb=snr_range[snr][0])
                    if init_dims['obstruct'] > 0 or init_dims['obstruct'] == -1:
                        if inc_flag:
                            env_dict['env_'+str(ii)] =  (env.src_coords,env.det_coords,env.intensity,env.bkg_intensity, env.obs_coord)
                            ii+=1
                            if (ii % 100) == 0:
                                print(f'Obs SNR: {np.round(snr_exp,3)} -> {counts}')
                    else:
                        if inc_flag:
                            env_dict['env_'+str(ii)] =  (env.src_coords,env.det_coords,env.intensity,env.bkg_intensity)
                            ii+=1
                            if (ii % 100) == 0:
                                print(f'SNR: {np.round(snr_exp,3)} -> {counts}')
                else:
                    env_dict['env_'+str(ii)] =  (env.src_coords,env.det_coords,env.intensity,env.bkg_intensity)
                    #print(f'Source coord: {env.src_coords}, Det coord: {env.det_coords}, Intensity: {env.intensity},{env.bkg_intensity}')
                    ii+=1
                    print(f'SNR: {np.round(snr_exp,3)}')
                
    #joblib.dump(env_dict, osp.join(save_path, 'test_env_dict_obs'+str(init_dims['obstruct'])+'_'+snr+'_v4'))

def load_env(random_ng, num_obs):
    import gym
    init_dims = {'bbox': [[0.0,0.0],[2700.0,0.0],[2700.0,2700.0],[0.0,2700.0]],
                'area_obs':[200.0,500.0], 
                'obstruct':num_obs, 'seed' : random_ng}
    env_name = 'gym_radloc:RadLoc-v0'
    env = gym.make(env_name,**init_dims)

    return env

def classify_snr(snr_exp,div,count,num_env,lb=0):
    inc = 0
    if count[0] < num_env and (lb < snr_exp <= (div * 1 + lb)):
        count[0] += 1
        inc = 1
    elif count[1] < (num_env) and ((div * 1 + lb) < snr_exp <= (div * 2 + lb)):
        count[1] += 1
        inc = 1
    elif count[2] < num_env and ((div * 2 + lb) < snr_exp <= (div * 3 + lb)):
        count[2] += 1
        inc = 1
    elif count[3] < num_env and ((div * 3 + lb) < snr_exp <= (div * 4 + lb)):
        count[3] += 1
        inc = 1
    #elif count[4] < num_env and (div * 4 < snr_exp):
    #    count[4] += 1
    #    inc = 1
    return count, inc

def set_vis_coord(point, coords):
    point.set_x(coords[0])
    point.set_y(coords[1])
    return point

def view_envs(path, max_obs, num_envs, render=True):
    
    for jj in range(1,max_obs):
        print(f'----------------Num_obs {jj} ------------------')
        rng = np.random.default_rng(robust_seed)
        env = load_env(rng,jj)
        _ = env.reset()
        env_set = joblib.load(path + str(jj))
        inter_count = 0
        repl = 0
        for kk in range(num_envs):
            _, env = refresh_env(env_set,env,kk,num_obs=len(env.obs_coord))
            L = vis.Line_Segment(env.detector, env.source)
            inter=False
            zz = 0
            while not inter and zz < jj:
                if vis.boundary_distance(L,env.poly[zz]) < 0.001:
                    inter = True
                    inter_count += 1
                zz += 1
                
            if render and repl < 5:
                fig, ax1 = plt.subplots(1,figsize=(5, 5),tight_layout=True)
                ax1.scatter(env.src_coords[0],env.src_coords[1],c='red',marker='*')
                ax1.scatter(env.det_coords[0],env.det_coords[1], c='black')
                ax1.grid()
                ax1.set_xlim(0,env.search_area[1][0])
                ax1.set_ylim(0,env.search_area[1][0])
                for coord in env.obs_coord:
                    p_disp = Polygon(coord[0])
                    ax1.add_patch(p_disp)
                plt.show()
                repl += 1
        print(f'Out of {num_envs} {inter_count/num_envs:2.2%} have an obstruction between source and detector starting position.')

def refresh_env(env_dict,env,n,num_obs=0):
    key = 'env_'+str(n)
    env.src_coords    = env_dict[key][0]
    env.det_coords    = env_dict[key][1]
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
    return o, env

if __name__ == '__main__':
    num_envs = 1000
    num_obs = 1
    seed = 500
    robust_seed = _int_list_from_bigint(hash_seed(seed))[0]
    rng = np.random.default_rng(robust_seed)
    init_dims = {'bbox': [[0.0,0.0],[2700.0,0.0],[2700.0,2700.0],[0.0,2700.0]],
            'area_obs':[200.0,500.0], 
            'obstruct':num_obs, 'seed' : rng}

    env_name = 'gym_rad_search:RadSearch-v0'
    save_p = '../snr/'
    load_p = '../snr/'
    #view_envs(load_p,num_obs, num_envs)
    #create_envs(num_envs, init_dims, env_name, save_p)
    create_envs_snr(num_envs, init_dims, env_name, save_p,snr='low')
    print('Saving... Done')
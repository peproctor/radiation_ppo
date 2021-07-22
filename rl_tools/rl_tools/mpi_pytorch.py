import multiprocessing
import numpy as np
import os
import torch
import time
from mpi4py import MPI
from rl_tools.mpi_tools import broadcast, mpi_avg, num_procs, proc_id, mpi_sum

def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    #print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads()==1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    #print('Proc %d: Reporting new number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)

def mpi_avg_grads(module):
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs()==1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()   # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]

def mpi_avg_params(module):
    if num_procs()==1:
        return
    for p in module.parameters():
        p_numpy = p.detach().numpy()   # numpy view of tensor data
        avg_p_grad = mpi_avg(p_numpy)
        p_numpy[:] = avg_p_grad[:]

def sync_params(module):
    """ Sync all parameters of module across all MPI processes. """
    if num_procs()==1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy) 

def sync_params_env(env_dict):
    if num_procs()==1:
        return env_dict
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # automatic MPI datatype discovery
        if rank == 0:
            data = env_dict #np.array([module.source.coords[0],module.source.coords[1],module.intensity,module.bkg_intensity],dtype=np.float64)
        else:
            data = None

        return comm.bcast(data,root=0)

def running_stats(obs,buff,count):
    mu_n = buff.mu_obs + (obs - buff.mu_obs)/(count)
    s_n = buff.sigma_obs + (obs - buff.mu_obs)*(obs-mu_n)
    buff.mu_obs = mu_n
    buff.sigma_obs = s_n
    #print(f'count: {count}')
    return buff.mu_obs, buff.sigma_obs

def sync_params_stats(x, t, stat_obj, count):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
    """
    if num_procs()==1 or t==-2:
        return running_stats(x,stat_obj,count)
    
    if t == -1:
        x = np.array(x, dtype=np.float32)
        global_sum, global_n = mpi_sum([x]), num_procs()
        global_mu = global_sum / global_n

        global_sum_sq = mpi_sum((x - global_mu)**2)
        global_sig = np.sqrt(global_sum_sq / global_n)  # compute global std
        global_sig[global_sig==0] = 1
        #print(f'Inside: Proc id: {proc_id()} -> Obs: {x}, Mean: {global_mu}, Std: {global_sig}')
        print(f'Inside: Proc id: {proc_id()} -> t=0')
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        mu, sig = np.array(stat_obj.mu_obs, dtype=np.float32), np.array(stat_obj.sigma_obs, dtype=np.float32)
        count = np.array(count, dtype=np.float32)
        
        x = np.append(x, count)
        print(f'Proc id: {proc_id()} -> Before Call Sum {t}')
        sum_op = mpi_sum([x]).squeeze()
        print(f'Proc id: {proc_id()} -> After Call Sum {t}')
        #print(f'Proc id: {proc_id()} -> Cat: {x}, Sum_op: {sum_op}')
        global_sum, glob_count = sum_op[0:3], sum_op[3]
        mean_obs = global_sum / num_procs()

        global_mu = mu + (mean_obs - mu)/glob_count
        global_sig = np.sqrt(sig + (mean_obs - mu)*(mean_obs - global_mu))
    #print(f'Inside: Proc id: {proc_id()} -> Mean: {global_mu}, Std: {global_sig}')
    return global_mu.squeeze(), global_sig.squeeze()

def synchronize():
    if num_procs() == 1:
        return
    print(f'Proc id: {proc_id()} -> BARRIER',flush=True)
    comm = MPI.COMM_WORLD
    comm.Barrier()

def update_stat_buff(stat_buff, stat_buff_act):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    root = None
    if not stat_buff and stat_buff_act:
        data = np.array([1],dtype=int)
        root = rank
        print(f'Proc id: {proc_id()} -> Broadcast {data} to stat buff')
    else:
        data = np.array([0], dtype=int)
    comm.Bcast(data, root=0)
    return data[0], root

    

def plot_mu(x_mu,x_sig):
    comm = MPI.REQUEST
    rank = comm.Get_rank()

    if rank == 0:
        #print('Plot mu',x_mu)
        #import sys
        #sys.exit()
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        plt.plot(range(len(x_mu)),x_mu)
        plt.title('Mean')
        plt.subplot(122)
        plt.plot(range(len(x_sig)),x_sig)
        plt.title('Std')
        plt.show()
        
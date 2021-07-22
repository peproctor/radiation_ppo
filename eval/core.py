import numpy as np
from numpy.linalg import inv
#import scipy.signal
from numba import jit
#from scipy.optimize import minimize
#from autograd import jacobian, hessian
#from autograd.numpy import sqrt
#from autograd.scipy.stats import poisson, norm
from scipy.stats import poisson
import math
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import joblib
import copy
#from multiprocessing import Pool



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity, layer_norm=False):
    layers = []

    if layer_norm:
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-1 else output_activation
            ln = nn.LayerNorm(sizes[j+1]) if j < len(sizes)-1 else None
            layers += [nn.Linear(sizes[j], sizes[j+1]),ln , act()]
        if None in layers:
            layers.remove(None)
    else:
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-1 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])



class StatBuff():
    def __init__(self):
        self.mu = 0
        self.sig_sto = 0
        self.sig_obs = 1
        self.count = 0

    def update(self,obs):
        self.count += 1
        if self.count == 1:
            self.mu = obs
        else:
            mu_n = self.mu + (obs - self.mu)/(self.count)
            s_n = self.sig_sto + (obs - self.mu)*(obs-mu_n)
            self.mu = mu_n
            self.sig_sto = s_n 
            self.sig_obs = math.sqrt(s_n /(self.count-1))
            #if len(self.sig_obs.shape) > 1:
            #    self.sig_obs[self.sig_obs==0] = 1
            #else:
            if self.sig_obs == 0:
                self.sig_obs = 1
        

    def reset(self):
        self.mu = 0
        self.sig_sto = 0
        self.sig_obs =  1
        self.count = 0

class LocMod(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, ln_preact=True, weight_init=False):
        super(LocMod,self).__init__()

        self.hs = 24#hidden_size[0] 
        self.loc_model = PFRNN_Policy(3,self.hs,1)
        self.loc_pred = mlp([self.hs] +[16] + [2], nn.Tanh)
        self.loc_pred = torch.nn.Sequential(*(list(self.loc_pred.children())[:-1]))
        #self.loc_input = nn.Linear(hidden_size[0],1)
        self.loc_tanh = nn.Tanh()
        

    def forward(self, x, hidden=None, ep_form=None, pred=False):
        hidden = self.loc_model(x[:,:,:3],hidden.view(1,1,self.hs))[0]
        #x = self.loc_tanh(self.loc_input(hidden))
        loc_pred = self.loc_tanh(self.loc_pred(hidden))
        return loc_pred, hidden

class PFRNNBaseCell(nn.Module):
    """parent class for PFRNNs
    """
    def __init__(self, num_particles, input_size, hidden_size, resamp_alpha,
            use_resampling, activation):
        """init function
        
        Arguments:
            num_particles {int} -- number of particles
            input_size {int} -- input size
            hidden_size {int} -- particle vector length
            resamp_alpha {float} -- alpha value for soft-resampling
            use_resampling {bool} -- whether to use soft-resampling
            activation {str} -- activation function to use
        """
        super(PFRNNBaseCell, self).__init__()
        self.num_particles = num_particles
        self.samp_thresh = num_particles * 1.0
        self.input_size = input_size
        self.h_dim = hidden_size
        self.resamp_alpha = resamp_alpha
        self.use_resampling = use_resampling
        self.activation = activation
        self.initialize = 'rand'
        #self.batch_norm = nn.BatchNorm1d(self.num_particles, track_running_stats=False)

    def resampling(self, particles, prob):
        """soft-resampling
        
        Arguments:
            particles {tensor} -- the latent particles
            prob {tensor} -- particle weights
        
        Returns:
            tuple -- particles
        """

        resamp_prob = self.resamp_alpha * torch.exp(prob) + (1 -
                self.resamp_alpha) * 1 / self.num_particles
        resamp_prob = resamp_prob.view(self.num_particles, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1),
                num_samples=self.num_particles, replacement=True)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        flatten_indices = indices.view(-1, 1).squeeze()

        # PFLSTM
        if type(particles) == tuple:
            particles_new = (particles[0][flatten_indices],
                    particles[1][flatten_indices])
        # PFGRU
        else:
            particles_new = particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 -
            self.resamp_alpha) / self.num_particles)
        prob_new = torch.log(prob_new).view(self.num_particles, -1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)

        return particles_new, prob_new

    def reparameterize(self, mu, var):
        """Implements the reparameterization trick introduced in [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
        
        Arguments:
            mu {tensor} -- learned mean
            var {tensor} -- learned variance
        
        Returns:
            tensor -- sample
        """
        std = torch.nn.functional.softplus(var)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.shape).normal_()
        else:
            eps = torch.FloatTensor(std.shape).normal_()

        return mu + eps * std


class PFGRUCell(PFRNNBaseCell):
    def __init__(self, num_particles, input_size, obs_size, hidden_size, resamp_alpha, use_resampling, activation):
        super().__init__(num_particles, input_size, hidden_size, resamp_alpha,
                use_resampling, activation)

        self.fc_z = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_r = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_n = nn.Linear(self.h_dim + self.input_size, self.h_dim * 2)

        self.fc_obs = nn.Linear(self.h_dim + self.input_size, 1)
        self.hid_obs = mlp([self.h_dim] + [24] + [2],nn.ReLU)
        self.hnn_dropout = nn.Dropout(p=0)

    def forward(self, input_, hx):
        """One step forward for PFGRU
        
        Arguments:
            input_ {tensor} -- the input tensor
            hx {tuple} -- previous hidden state (particles, weights)
        
        Returns:
            tuple -- new tensor
        """

        h0, p0 = hx
        obs_in = input_.repeat(h0.shape[0],1)
        obs_cat= torch.cat((h0, obs_in), dim=1)

        z = torch.sigmoid(self.fc_z(obs_cat))
        r = torch.sigmoid(self.fc_r(obs_cat))
        n = self.fc_n(torch.cat((r * h0, obs_in), dim=1))

        mu_n, var_n = torch.split(n, split_size_or_sections=self.h_dim, dim=1)
        n = self.reparameterize(mu_n, var_n)

        if self.activation == 'relu':
            # if we use relu as the activation, batch norm is require
            n = n.view(self.num_particles, -1, self.h_dim).transpose(0,
                    1).contiguous()
            n = self.batch_norm(n)
            n = n.transpose(0, 1).contiguous().view(-1, self.h_dim)
            n = torch.relu(n)
        elif self.activation == 'tanh':
            n = torch.tanh(n)
        else:
            raise ModuleNotFoundError

        h1 = (1 - z) * n + z * h0

        p1 = self.observation_likelihood(h1, obs_in, p0)

        if self.use_resampling: #and (1/(p1.exp().square().sum())) < (self.samp_thresh):
            h1, p1 = self.resampling(h1, p1)

        p1 = p1.view(-1, 1) 

        mean_hid = (torch.exp(p1) * self.hnn_dropout(h1)).sum(axis=0)
        loc_pred = self.hid_obs(mean_hid)

        return loc_pred, (h1, p1)
    

    def observation_likelihood(self, h1, obs_in, p0):
        """observation function based on compatibility function
        """
        logpdf_obs = self.fc_obs(torch.cat((h1, obs_in), dim=1))

        p1 = logpdf_obs + p0

        p1 = p1.view(self.num_particles, -1, 1)
        p1 = nn.functional.log_softmax(p1, dim=0)

        return p1

    def init_hidden(self, batch_size):
        initializer = torch.rand if self.initialize == 'rand' else torch.zeros
        h0 = initializer(batch_size * self.num_particles, self.h_dim)
        p0 = torch.ones(batch_size * self.num_particles, 1) * np.log(1 / self.num_particles)
        hidden = (h0, p0)
        return hidden

class SeqLoc(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, ln_preact=True, weight_init=False):
        super(SeqLoc,self).__init__()
        
        self.seq_model = nn.GRU(input_size,hidden_size[0][0],1)#nn.RNN(input_size,hidden_size[0],1)
        self.Woms = mlp(hidden_size[0] + hidden_size[1] + [2],nn.Identity) #nn.Linear(hidden_size[0], 4, bias=True)
        self.Woms = torch.nn.Sequential(*(list(self.Woms.children())[:-1]))

        if weight_init:
            for m in self.named_children():
                self.weights_init(m)

        #self.mlp = mlp(hidden_size+[1],nn.Tanh)
        self.hs = hidden_size[0][0]
    
    def weights_init(self,m):
        if isinstance(m[1], nn.Linear):
            stdv = 2 / math.sqrt(max(m[1].weight.size()))
            m[1].weight.data.uniform_(-stdv, stdv)
            if m[1].bias is not None:
                m[1].bias.data.uniform_(-stdv, stdv)
 
    
    def forward(self, x, hidden=None, ep_form=None, batch=False): #MS POMDP
        if not(batch):
            hidden = self.seq_model(x.unsqueeze(0),hidden)[0]
        else:
            hidden = self.seq_model(x.unsqueeze(1),hidden)[0]
        out_arr = self.Woms(hidden.squeeze())
        return out_arr, hidden

         
    def init_hidden(self,bs):
        std = 1.0 / math.sqrt(self.hs)
        init_weights = torch.FloatTensor(1,1,self.hs).uniform_(-std,std)
        return init_weights[0,:,None]

class SeqPt(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, ln_preact=True, weight_init=False):
        super(SeqPt,self).__init__()
        
        self.seq_model = nn.GRU(input_size,hidden_size[0],1)#nn.RNN(input_size,hidden_size[0],1)
        self.Woms = mlp([hidden_size[0]] + [hidden_size[1]] + [8],nn.Tanh) #nn.Linear(hidden_size[0], 4, bias=True)
        self.Woms = torch.nn.Sequential(*(list(self.Woms.children())[:-1]))
        self.Valms = mlp([hidden_size[0]] + [hidden_size[2]] + [1], nn.Tanh)
        self.Valms = torch.nn.Sequential(*(list(self.Valms.children())[:-1]))

        if weight_init:
            for m in self.named_children():
                self.weights_init(m)

        #self.mlp = mlp(hidden_size+[1],nn.Tanh)
        self.hs = hidden_size[0]
    
    def weights_init(self,m):
        if isinstance(m[1], nn.Linear):
            stdv = 2 / math.sqrt(max(m[1].weight.size()))
            m[1].weight.data.uniform_(-stdv, stdv)
            if m[1].bias is not None:
                m[1].bias.data.uniform_(-stdv, stdv)
 
    
    def forward(self, x, hidden=None, ep_form=None, pred=False): #MS POMDP
        #if x.size(-1) == 3:
        hidden = self.seq_model(x,hidden)[0]
        out_arr = self.Woms(hidden.squeeze())
        val = self.Valms(hidden.squeeze())
        return out_arr, hidden, val
        #else:    
        #    hidden = self.seq_model(x,hidden)
        #    out_arr = self.Woms(hidden)
        #    return out_arr, hidden
         
    def _reset_state(self):
        std = 1.0 / math.sqrt(self.hs)
        init_weights = torch.FloatTensor(1,self.hs).uniform_(-std,std)
        return (init_weights[0,:,None],0)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None, hidden=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        #pi, hidden = self._distribution(obs,hidden) #should be [4000,5]
        pi, hidden, val = self._distribution(obs,hidden=hidden)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act) #should be [4000]
        return pi, logp_a, hidden, val

class ExploreActor(Actor):
    def __init__(self, obs_dim,env=None):
        super().__init__()
        self.dim_area = obs_dim
        self.ACTION_LS = np.array([[-100,0],
                                   [0,100],
                                   [100,0],
                                   [0,-100]])
        self.env = env
        
    def _distribution(self, obs, hidden=None):
        score = np.infty
        det_coords = self.env.det_coords.copy()
        obs = obs.numpy()[0,0,7:] * self.env.search_area[1][0]
        for i, act in enumerate(self.ACTION_LS):
            J = np.linalg.norm(obs - (det_coords+act))
            if J < score:
                score = J
                optim = i
        return torch.tensor(optim), 0, torch.tensor(0)
        
    def update(self,env):
        self.env = env
    def _log_prob_from_distribution(self, pi, act):
        return torch.tensor(0)

    def _reset_state(self):
        return 0



class MLPCategoricalActor(Actor):
    
    def __init__(self, input_dim, act_dim, hidden_sizes, activation,net_type=None,batch_s=1):
        super().__init__()

        if net_type == 'rnn':
            self.logits_net = RecurrentNet(input_dim,act_dim, hidden_sizes, activation, batch_s=batch_s,rec_type='rnn')
        else:
            if hidden_sizes:
                self.logits_net = mlp([input_dim] + hidden_sizes + [act_dim], activation)
            else:
                self.logits_net = mlp([input_dim] + [act_dim], activation)

    def _distribution(self, obs, hidden=None):
        #logits, hidden = self.logits_net(obs, hidden=hidden)
        logits, hidden, val = self.logits_net.v_net(obs,hidden=hidden)
        if torch.any(torch.isnan(logits)):
            print(f'Logits {logits} ------> run {run}')
            logits[torch.isnan(logits)] = -50
        return Categorical(logits=logits), hidden, val

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def _reset_state(self):
        return self._get_init_states()

    def _get_init_states(self):
        std = 1.0 / math.sqrt(self.logits_net.hs)
        init_weights = torch.FloatTensor(1,1,self.logits_net.hs).uniform_(-std,std)
        return init_weights[0,:,None]

class RecurrentNet(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, batch_s=1,rec_type='lstm'):
        super().__init__()
        self.hs = hidden_sizes[0]
        self.v_net = SeqPt(obs_dim//batch_s, hidden_sizes)

    def forward(self, obs, hidden, ep_form=None, meas_arr=None):
        return self.v_net(obs, hidden, ep_form=ep_form)
        #return torch.squeeze(self.v_net(obs, hidden), -1) # Critical to ensure v has right shape.

    def _reset_state(self):
        return self._get_init_states()

    def _get_init_states(self):
        std = 1.0 / math.sqrt(self.v_net.hs)
        init_weights = torch.FloatTensor(2,self.v_net.hs).uniform_(-std,std)
        return (init_weights[0,:,None], init_weights[1,:,None])

class RNNModelActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden = (32,),
                 hidden_sizes_pol=(64,), hidden_sizes_val=(64,64), hidden_sizes_rec=(64,),
                 activation=nn.Tanh,net_type=None, pad_dim=2,batch_s=1, seed=0):
        super().__init__()
        self.seed_gen = torch.manual_seed(seed)
        obs_dim = observation_space.shape[0]# + pad_dim
        self.hidden = hidden[0]
        self.pi_hs = hidden_sizes_rec[0]
        self.val_hs = hidden_sizes_val[0]
        self.bpf_hsize = hidden_sizes_rec[0]
        hidden_sizes = hidden + hidden_sizes_pol + hidden_sizes_val

        if hidden_sizes_pol[0] == 1:
            self.pi = MLPCategoricalActor(self.pi_hs, action_space.n, None, activation, net_type=net_type,batch_s=batch_s)
        else:
            self.pi = MLPCategoricalActor(obs_dim + pad_dim , action_space.n, hidden_sizes, activation, net_type=net_type,batch_s=batch_s)

        # build value function
        #self.v  = MLPCritic(self.pi_hs, hidden_sizes_val[0], activation)
        self.num_particles = 40
        self.alpha = 0.7
        
        #self.model   = SeqLoc(obs_dim-8,[hidden_sizes_rec]+[[24]],1)
        self.model  = PFGRUCell(self.num_particles,obs_dim-8,obs_dim-8,self.bpf_hsize,self.alpha,True, 'tanh') #obs_dim, hidden_sizes_pol[0]
    
    def set_seed(self,seed):
        self.seed = seed
        self.seed_gen = torch.manual_seed(seed)

    def step(self, obs, hidden=None):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            loc_pred, hidden_part = self.model(obs_t[:,:3], hidden[0])
            obs_t = torch.cat((obs_t,loc_pred.unsqueeze(0)),dim=1)
            pi, hidden2, v  = self.pi._distribution(obs_t.unsqueeze(0), hidden[1])
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            hidden = (hidden_part,hidden2) 
        return a.numpy(), v.numpy(), logp_a.numpy(), hidden, loc_pred.numpy()

    def grad_step(self, obs,act, hidden=None):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(1)
        loc_pred = torch.empty((obs_t.shape[0],2))
        hidden_part = hidden[0]
        with torch.no_grad():
            for kk, o in enumerate(obs_t):
                loc_pred[kk], hidden_part = self.model(o[:,:3], hidden_part)
        obs_t = torch.cat((obs_t,loc_pred.unsqueeze(1)),dim=2)
        pi, logp_a, hidden2, val  = self.pi(obs_t, act=act, hidden=hidden[1])
        return pi, val, logp_a, loc_pred

    def act(self, obs, hidden=None):
        return self.step(obs,hidden=hidden)

    def reset_hidden(self, batch_size=1):
        model_hidden = self.model.init_hidden(batch_size)
        a2c_hidden = self.pi._reset_state()
        return (model_hidden, a2c_hidden)

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes_pol=(64,), hidden_sizes_val=(64,64),
                 activation=nn.Tanh,lstm=None, pad_dim=0,batch_s=1):
        super().__init__()

        obs_dim = observation_space.shape[0] + pad_dim
        self.pi_hs = hidden_sizes_pol[0]

        self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes_pol, activation, lstm=lstm,batch_s=batch_s)
        self.v  = RecurrentNet(obs_dim, action_space.n, hidden_sizes_val[0], activation, batch_s=batch_s,rec_type='rnn')
        
    def step(self, obs, hidden=None):
        with torch.no_grad():
            #pi, hidden = self.pi._distribution(obs, hidden=hidden)
            obs_t = torch.as_tensor(obs[None], dtype=torch.float32)
            v, hidden = self.v(obs_t,hidden)
            #obs_t = torch.cat((obs_t.squeeze(),hidden[0].squeeze())).view(1,self.pi_hs)
            pi = self.pi._distribution(hidden)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            #v = self.v(obs)   
        return a.numpy(), v.numpy(), logp_a.numpy(), hidden

    def act(self, obs):
        return self.step(obs)[0]

class ParticleFilter(nn.Module):
    def __init__(self, s_size=3, nParticles=1000, noise_params=[1,1], bkg=15, vel=2, rng=None, k=None,
                 intensity=[1e2,1e3],coord=[0,25e2],thresh=0.1,kern_width=0.8,kern='biwe',L=None,scale=None,
                 alpha=None, interval=None, fim_thresh=None):
        super(ParticleFilter, self).__init__()
        self.state_dim = s_size
        self.nPart = nParticles
        self.xp = np.zeros((self.nPart,s_size))
        self.wp = np.zeros((self.nPart,1))
        self.nEff = np.array([])
        self.v = vel
        self.xpHatMean = np.zeros(s_size)
        self.proSigma  = np.array([noise_params[1],noise_params[0],noise_params[0]])
        self.processNoise = np.zeros((self.nPart,self.state_dim)) 
        self.bkg = bkg
        self.rThresh = thresh
        self.nEffThresh = thresh * self.nPart
        self.coord_bound = coord
        self.int_bound = intensity
        self.n = 0
        self.k = k
        self.N_dim = (nParticles)**(-1/s_size)
        self.rng = rng
        self.prev_det = np.array([])
        self.ACTION_LS  = np.array([0,1,2,3,4,5,6,7])
        self.kern_width = [50000,1.2,1.2]
    def track(self,meas,a=None):
        if self.n == 0:
            self.xp[:,0]   = self.rng.uniform(self.int_bound[0],self.int_bound[1],size=(self.nPart))
            self.xp[:,1:]  = self.rng.uniform(self.coord_bound[0],self.coord_bound[1],size=(self.nPart,self.state_dim-1))
            self.wp[:]     = 1/self.nPart
            self.xp_init   = self.xp.copy()
            self.wp_init   = self.wp.copy()
            self.wp[:]     = np.log(self.wp[:])
        else:
            self.processNoise = self.rng.normal(0,self.proSigma, size=(self.nPart,self.state_dim))
            self.processModel(self.processNoise,a)

        self.poisson_ll(meas)

        #Reweight particles
        self.wp = np.exp(self.wp - self.wp.max())
        self.wp = self.wp / self.wp.sum()
        self.nEff = np.append(self.nEff,np.round(1/np.sum(np.square(self.wp))))

        if (self.nEff[self.n] < self.nEffThresh):
            u = self.rng.uniform(size=self.nPart-1) #np.random.rand(self.nPart - 1)
            self.xp = self.xp[ssp(self.wp,self.nPart,u)]
            self.wp[:] = 0
            self.poisson_ll(meas)
            self.wp = np.exp(self.wp - self.wp.max())
            self.wp = self.wp / self.wp.sum()
            
        self.update_measures()
        if self.n == 0:
            self.xp_prev   = self.xp[:,None,:].copy()
            self.wp_prev   = self.wp[:,None,:].copy()
        elif self.n > 0:
            self.xp_prev = np.hstack((self.xp_prev,self.xp[:,None,:]))
            self.wp_prev = np.hstack((self.wp_prev,self.wp[:,None,:]))
        self.wp = np.log(self.wp)
        self.n += 1

        return self.xpHatMean

    
    def measModel(self,x_det):
        R = np.square(np.linalg.norm((self.xp[:,1:]-x_det),axis=1))
        return (np.round(self.xp[:,0]*1e4 / R) + self.bkg).squeeze()
        #return ((self.xp[:,0] / R)).squeeze()
    def processModel(self,noise,a):
        #self.xp[:,3:]  = self.xp[:,1:3] + self.ACTION_LS[a] + noise[:,1:3]
        self.xp[:,1:] = self.xp[:,1:] + noise[:,1:]
        self.xp[:,0]   = np.clip(self.xp[:,0] + noise[:,0], 0, np.infty)
        
    def poisson_ll(self,meas,log=True):
        if log:
            lam = self.measModel(meas[1:])
            ll_hood = poisson._logpmf(meas[0],lam)[:,None]
            self.wp = (self.wp + ll_hood)
        else:
            lam = np.repeat(self.measModel(meas[0])[:,None],meas[1].shape[0],axis=1)
            return poisson._pmf(meas[1],lam)
            

    def update_measures(self):
        #self.xpHatMean = np.sum(np.multiply(np.exp(self.wp),self.xp),axis=0)
        self.xpHatMean = np.sum(np.multiply(self.wp,self.xp),axis=0)

    def init_hidden(self,bs):
        return None
            


class GradSearch(object):
    def __init__(self,q=1,env=None):
        self.q_rec = (1/q)
        self.env = env
        self.ACTION_LS = np.array([0,1,2,3,4,5,6,7])
        self.softmax = nn.Softmax(0)
        self.grad = torch.zeros(len(self.ACTION_LS))
        self.probs = torch.zeros_like(self.grad)
        self.pointer = 0

    def update(self,env):
        self.env = env
        self.pointer = 0 

    def step(self,obs):
        det_coords = self.env.det_coords.copy()
        self.pointer += 1
        for act in self.ACTION_LS:
            o,_,_,_ = self.env.step(act)
            if (o[1:3] == obs[1:3]).all():
                self.grad[act] = 0
            else:
                self.grad[act] = (o[0]-obs[0])*0.01*self.q_rec
                self.env.det_coords = det_coords.copy()
                self.env.detector.set_x(det_coords[0])
                self.env.detector.set_y(det_coords[1])
            
        probs = Categorical(self.softmax(self.grad))
        self.env.det_sto = self.env.det_sto[:self.pointer]
        self.env.meas_sto = self.env.meas_sto[:self.pointer]
        self.env.iter_count = self.pointer
        return probs.sample().item()

class FIC(nn.Module):
    def __init__(self, s_size=3, nParticles=1000, noise_params=[1,1], bkg=1, rng=None, det_step=100, 
                 intensity=[1e6,10e6],coord=[0,25e2],thresh=0.1,L=1,vel=100, FIM_step=None,k=None,
                 scale=None, interval=[100,100], alpha=0.5, r_div=1, fim_thresh=0.4):
        super(FIC, self).__init__()
        self.nPart = nParticles
        self.rng = rng
        self.pro_sigs = noise_params
        self.bkg = bkg
        self.thresh = thresh
        self.L = L
        self.det_step = det_step
        self.s_size = s_size
        self.bpf = ParticleFilter(self.s_size, self.nPart, self.pro_sigs, self.bkg, self.det_step, rng,k=k, thresh=self.thresh)
        self.ACTION_LS  = np.array([0,1,2,3,4,5,6,7])
        self.FIM_step = FIM_step
        self.scale = scale
        self.interval = interval
        self.alpha = alpha
        self.RDIV_FLAG = r_div
        self.fim_thr = fim_thresh

    def FIM(self,x,x_s,I):
        I = I * 1e4
        denom = sum((x - x_s)**2)
        grad_xy = 2*(x - x_s)*I / (denom**2)
        grad_I = 1 / denom
        grad = np.append(grad_I, grad_xy)
        J = np.outer(grad,grad)/(I/denom + self.bkg)
        return J@self.scale

    def particle_FIM(self,x_det,x_s,wp_prev,s_size):
        pred = x_s.copy()
        pred[:,0] = pred[:,0]*1e4
        denom = np.sum(np.square(x_det - pred[:,1:]),axis=1)
        grad_xy = (2*(x_det - pred[:,1:]))*(pred[:,0] / np.square(denom))[:,None]
        grad_I = 1 / denom
        grad = np.hstack((grad_I[:,None], grad_xy))
        J = np.einsum('ij,ikl->ijk',grad,grad[:,:,None])* np.tile((1/(pred[:,0]/denom + self.bkg))[:,None,None],(s_size,s_size))
        return ((J@self.scale)*wp_prev[:,None]).sum(axis=0)

    def renyi_div(self,z,x_det,x_s,wp_prev):
        l_hood = self.bpf.poisson_ll([x_det,z],log=False)
        p_z = (wp_prev * l_hood).sum(axis=0)
        p_z_a = (wp_prev * (l_hood**self.alpha)).sum(axis=0)
        r_div = (1/(self.alpha-1))*((p_z *(np.log(p_z_a) - self.alpha*np.log(p_z))).sum())
        return r_div

    def optim_action(self,x,x_hat,step):
        num_act = len(self.ACTION_LS)
        x_act = [[] for _ in range(num_act+1)] #np.zeros((L,4,2))
        J = np.zeros((num_act))
        J_fish = np.zeros((num_act))
        meas_dis = np.arange(np.clip(x[0]-self.interval[0],1,np.infty),x[0]+self.interval[1],1)
        for act in self.ACTION_LS:
            #L=1
            l = 0
            i = 0
            j = 0
            m = 0
            lay = 1
            x_act_1 = self.FIM_step(act)
            if self.RDIV_FLAG:
                J[act] = self.renyi_div(meas_dis,x_act_1,self.bpf.xp_prev[:,step,:],
                                    self.bpf.wp_prev[:,step,:])
                J_fish[act] = np.trace(self.particle_FIM(x_act_1,self.bpf.xp_prev[:,step,:],
                                  self.bpf.wp_prev[:,step,:],self.s_size))
            else:
                J[act] = np.trace(self.particle_FIM(x_act_1,self.bpf.xp_prev[:,step,:],
                                  self.bpf.wp_prev[:,step,:],self.s_size))
                J_fish[act] = J[act]
            
            x_act_2 = copy.deepcopy(x_act)
            while l < (self.L-1):
                if l == 0:
                    #L=2
                    for act_2 in self.ACTION_LS:
                        x_act_2[l].append(self.FIM_step(act_2,x_act_1))
                        J[act] += np.trace(np.abs(inv(self.particle_FIM(x_act_2[l][act_2],self.bpf.xp_prev[:,step,:],
                                            self.bpf.wp_prev[:,step,:],self.s_size))))
                    l += 1
                else:
                    for act_2 in self.ACTION_LS:
                        if j < (num_act+1):
                            #L=3
                            x_act_2[j].append(self.FIM_step(act_2,x_act_2[l-1][i]))
                            J[act] += np.trace(np.abs(inv(self.particle_FIM(x_act_2[j][act_2],self.bpf.xp_prev[:,step,:],
                                                self.bpf.wp_prev[:,step,:],self.s_size))))
                        else:
                            #L=4
                            J[act] += np.trace(np.abs(inv(self.particle_FIM(FIM_step(act_2,x_act_2[lay][i]),self.bpf.xp_prev[:,step,:],
                                                self.bpf.wp_prev[:,step,:],self.s_size))))
                            m+=1
                    i += 1
                if j % 4 == 0 and j > 0:
                    i = 0
                    if m % 16 == 0 and m != 0:
                        lay += 1
                        m = 0
                    if lay % 4 == 0 or lay == 1:
                        l += 1
                j+=1

        if self.RDIV_FLAG == 1 and J.max() > self.fim_thr:
            self.RDIV_FLAG = 0
            action = J.argmax()
        else:
            action = J.argmax()

        return action, J_fish[action]


@jit(nopython=True)
def ssp(W, M, u):
    N = W.shape[0]
    MW = M * W
    nr_children = np.floor(MW).astype(np.int64)
    xi = MW - nr_children
    i, j = 0, 1
    for k in range(N - 1):
        delta_i = np.minimum(xi[j], 1. - xi[i])[0]  # increase i, decr j
        delta_j = np.minimum(xi[i], 1. - xi[j])[0]  # the opposite
        sum_delta = delta_i + delta_j
        # prob we increase xi[i], decrease xi[j]
        pj = delta_i / sum_delta if sum_delta > 0. else 0. 
        # sum_delta = 0. => xi[i] = xi[j] = 0.
        if u[k] < pj:  # swap i, j, so that we always inc i
            j, i = i, j
            delta_i = delta_j
        if xi[j] < 1. - xi[i]:
            xi[i] += delta_i
            j = k + 2
        else:
            xi[j] -= delta_i
            nr_children[i] += 1
            i = k + 2
    # due to round-off error accumulation, we may be missing one particle
    if np.sum(nr_children) == M - 1:
        last_ij = i if j == k + 2 else j
        if xi[last_ij] > 0.99:
            nr_children[last_ij] += 1
    if np.sum(nr_children) != M:
        # file a bug report with the vector of weights that causes this
        raise ValueError('ssp resampling: wrong size for output, file a bug report')
    return np.arange(N).repeat(nr_children[:,0])

    
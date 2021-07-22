import numpy as np
import scipy.signal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

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


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

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
            if self.sig_obs == 0:
                self.sig_obs = 1
        
    def reset(self):
        self.mu = 0
        self.sig_sto = 0
        self.sig_obs =  1
        self.count = 0

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
        if activation == 'relu':
             self.batch_norm = nn.BatchNorm1d(self.num_particles, track_running_stats=False)

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

        if self.use_resampling:
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
        
        self.seq_model = nn.GRU(input_size,hidden_size[0][0],1)
        self.Woms = mlp(hidden_size[0] + hidden_size[1] + [2],nn.Tanh) 
        self.Woms = torch.nn.Sequential(*(list(self.Woms.children())[:-1]))

        if weight_init:
            for m in self.named_children():
                self.weights_init(m)

        self.hs = hidden_size[0][0]
    
    def weights_init(self,m):
        if isinstance(m[1], nn.Linear):
            stdv = 2 / math.sqrt(max(m[1].weight.size()))
            m[1].weight.data.uniform_(-stdv, stdv)
            if m[1].bias is not None:
                m[1].bias.data.uniform_(-stdv, stdv)
 
    
    def forward(self, x, hidden=None, ep_form=None, batch=False): 
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
        
        self.seq_model = nn.GRU(input_size,hidden_size[0][0],1)
        self.Woms = mlp(hidden_size[0] + hidden_size[1] + [8],nn.Tanh)
        self.Woms = torch.nn.Sequential(*(list(self.Woms.children())[:-1]))
        self.Valms = mlp(hidden_size[0] + hidden_size[2] + [1], nn.Tanh)
        self.Valms = torch.nn.Sequential(*(list(self.Valms.children())[:-1]))

        if weight_init:
            for m in self.named_children():
                self.weights_init(m)

        self.hs = hidden_size[0]
    
    def weights_init(self,m):
        if isinstance(m[1], nn.Linear):
            stdv = 2 / math.sqrt(max(m[1].weight.size()))
            m[1].weight.data.uniform_(-stdv, stdv)
            if m[1].bias is not None:
                m[1].bias.data.uniform_(-stdv, stdv)
 
    
    def forward(self, x, hidden=None, ep_form=None, pred=False): #MS POMDP
        hidden = self.seq_model(x,hidden)[0]
        out_arr = self.Woms(hidden.squeeze())
        val = self.Valms(hidden.squeeze())
        return out_arr, hidden, val

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
        self.hs = hidden_sizes[0][0]
        self.v_net = SeqPt(obs_dim//batch_s, hidden_sizes)

    def forward(self, obs, hidden, ep_form=None, meas_arr=None):
        return self.v_net(obs, hidden, ep_form=ep_form)

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
        
        if hidden_sizes_pol[0][0] == 1:
            self.pi = MLPCategoricalActor(self.pi_hs, action_space.n, None, activation, net_type=net_type,batch_s=batch_s)
        else:
            self.pi = MLPCategoricalActor(obs_dim + pad_dim , action_space.n, hidden_sizes, activation, net_type=net_type,batch_s=batch_s)

        self.num_particles = 40
        self.alpha = 0.7
        
        #self.model   = SeqLoc(obs_dim-8,[hidden_sizes_rec]+[[24]],1)
        self.model  = PFGRUCell(self.num_particles,obs_dim-8,obs_dim-8,self.bpf_hsize,self.alpha,True, 'tanh') #obs_dim, hidden_sizes_pol[0]

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


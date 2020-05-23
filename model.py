import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch import optim
import numpy as np
from tqdm import trange

import logging
L = logging.getLogger(__name__)


def to_one_hot(indices, max_index):
  zeros = torch.zeros(
    indices.shape[0], max_index, dtype=torch.float32,
    device=indices.device)
  return zeros.scatter_(1, indices.unsqueeze(-1), 1)


def pairwise_distance(x, y):
  '''BxNxD, BxMxD -> BxNxM, dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2
  https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3'''
  x_norm = (x**2).sum(2).view(x.shape[0],x.shape[1],1)
  y_t = y.permute(0,2,1).contiguous()
  y_norm = (y**2).sum(2).view(y.shape[0],1,y.shape[1])
  dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
  dist[dist != dist] = 0 # replace nan values with 0
  return 0.5 * torch.clamp(dist, 0.0, np.inf)


def pairwise_softmax(z1, z2, temperature=1.0):
  '''Eq. (9,12). Returns (A, N, M) softmax of negative distance.'''
  A, N, D = z1.shape
  A, M, D = z2.shape
  return F.softmax(-pairwise_distance(z1, z2) / temperature, dim=-1)


class ValueIteration:
  def __init__(self, learner, prototype_states, backup=500):
    self.device = learner.device
    self.learner = learner

    #> we opt for a low value for n to assign most Q-value to closest state
    self.interpolation_param = 1e-20
    #> temperature is grid-searched [1.0, 0.1, 0.001, ..., 1e-20]
    self.trans_temperature = 1.0
    self.backup = backup

    action_dim = learner.action_dim
    self.actions = torch.arange(0, action_dim).to(self.device)
    self.prototype_states = self.learner.Z_theta(torch.tensor(
      prototype_states).to(self.device)).unique(dim=0)
    self.prototype_values = torch.zeros(len(self.prototype_states)).to(self.device)

    #L.debug('Uniq prototypes: %s', self.prototype_states.shape)

  def pi(self, state, valid_actions):
    '''Finds argmax. z*=Z(s) => argmax Q(Z(s),a) for a \in valid_actions.'''
    state = torch.tensor(state).to(self.device)
    valid_actions = torch.tensor(valid_actions).to(self.device)

    z = self.learner.Z_theta(state.unsqueeze(0))
    zs = z.unsqueeze(0).expand(valid_actions.shape[0], 1, z.shape[1])

    Q_zs_a = self.Q(zs, valid_actions)
    return valid_actions[Q_zs_a.argmax(dim=0)]

  def Q(self, zs, actions):
    '''Interpolate Q for each action based on nearest prototypes' Q-vals.
    Q(z*, a) = sum_x { w(z*, x) * Q(x, a) } where x are prototype states.'''
    w_zs_x = self.w(zs)
    # cached; could keep only proto values and recompute if memory is an issue.
    Q_x_a = self.qvals[actions].unsqueeze(1).expand_as(w_zs_x)
    return torch.sum(w_zs_x * Q_x_a, dim=-1) # A,Nz,Np -> A,Nz (interpolated Q)

  def w(self, zs):
    '''Eq. (12) softmax over prototypes. returns A,Nz,Np(softmax)'''
    A, Nz, D = zs.shape
    Np, D = self.prototype_states.shape
    x = self.prototype_states.unsqueeze(0).expand(A, Np, D)
    return pairwise_softmax(zs, x, temperature=self.interpolation_param)

  def closest(self, zs):
    '''Find indices of closest prototypes to `zs`. (A, Nz)'''
    A, Nz, D = zs.shape
    Np, D = self.prototype_states.shape
    x = self.prototype_states.unsqueeze(0).expand(A, Np, D)
    return pairwise_distance(zs, x).argmin(dim=-1)

  def plan(self, goal_state, discount=0.9, stop_iter_epsilon=1e-50):
    '''Add goal state to protos; plan using a discretized latent-space MDP.'''
    with torch.no_grad():
      goal_state = torch.tensor(goal_state).to(self.device).unsqueeze(0)
      z = self.learner.Z_theta(goal_state)
      idx = self.closest(z.unsqueeze(0)).item()

      already_proto = torch.max((z - self.prototype_states[idx])**2) < 1e-9
      if not already_proto.item():
        q = torch.tensor([0.]).to(self.device)
        self.prototype_states = torch.cat((self.prototype_states, z), dim=0)
        self.prototype_values = torch.cat((self.prototype_values, q), dim=0)
        idx = self.closest(z.unsqueeze(0)).item()  # "unit test"
        assert idx == len(self.prototype_values) - 1

      self.goal_idx = idx
      return self._plan(discount, stop_iter_epsilon)

  def _plan(self, discount=0.9, stop_iter_epsilon=1e-50):
    A, = self.actions.shape
    Np, D = self.prototype_states.shape

    qstates = self.prototype_states.unsqueeze(0).expand(A, Np, D)
    actions = self.actions.unsqueeze(1).expand(A, Np)

    # Discretize: find *closest prototype* for each predicted next latent state.
    next_states_predicted = self.learner.T(qstates, actions)
    self.next_idxs = pairwise_distance(next_states_predicted, qstates).argmin(-1)
    # Eqn. (9) == assume states connected in state space will also be in latent.
    # T(zj|zi,a) = softmax(-d(zj, zi+A(zi,a)) / t) where zi \in X
    # NOTE(tk) I would assume that high-reward states might be far away.
    T = pairwise_softmax(qstates, next_states_predicted, self.trans_temperature)
    T = T.gather(-1, self.next_idxs.unsqueeze(-1)).squeeze(-1)

    #> We use this reward function in planning, R(x)=1 if x=Z(sg) else 0
    reward = (self.next_idxs == self.goal_idx).float()

    for _ in range(self.backup):
      next_vals = self.prototype_values.repeat(A, 1).gather(-1, self.next_idxs)
      self.qvals = T*(reward+discount*next_vals)
      vnew, _ = self.qvals.max(dim=0)
      delta = (self.prototype_values - vnew).abs().max().item()
      self.prototype_values = vnew
      if delta < stop_iter_epsilon:
          break


class Learner(nn.Module):
  '''Uses notation from the paper mostly.'''
  def __init__(self, device, in_shape=(3, 60, 60), latent_dim=50,
      negative_samples=1, action_dim=4):
    super().__init__()

    self.action_dim = action_dim
    self.device = device
    self.J = negative_samples
    self.hinge = 1.0
    self.latent_dim = latent_dim

    self.A = LatentToTrans(
        latent_dim=self.latent_dim,
        action_dim=self.action_dim).to(device)
    self.R = LatentToReward(self.latent_dim).to(device)
    self.Z_theta = ObservationToLatent(
        latent_dim=self.latent_dim,
        in_shape=in_shape).to(device)

  def T(self, zs, act):
    '''Transition model.'''
    return zs + self.A(zs, act)

  def loss(self, obs, act, reward, obs_next):
    B = obs.shape[0]

    # B x latent_dim
    zs = self.Z_theta(obs)
    zs_next = self.Z_theta(obs_next)
    zs_next_pred = self.T(zs, act)

    distance = lambda a, b: 0.5 * torch.sum((a - b) ** 2, dim=-1)

    # B <<scalar>>
    # transition->map vs. map->transition
    loss_T = distance(zs_next_pred, zs_next)

    # B - B <<scalar>>
    # reward distance in latent space vs. original MDP
    loss_R = distance(reward, self.R(zs).flatten())

    # Paper says J=1 is fine too when reward loss is taken into account.
    zs_neg = zs.repeat(self.J, 1)[np.random.permutation(B*self.J)]
    zs_next_pred = zs_next_pred.repeat(self.J, 1)
    zeros = torch.zeros(B*self.J).to(self.device)
    loss_neg = torch.max(zeros, self.hinge - distance(zs_neg, zs_next_pred))

    return (loss_T.sum() + loss_R + loss_neg.sum()) / B


class ObservationToLatent(nn.Module):
  def __init__(self, latent_dim, in_shape):
    super().__init__()

    in_channels, w, h = in_shape

    self.cnn1 = nn.Sequential(
      nn.Conv2d(in_channels, 16, (3,3), padding=1),
      nn.ReLU())
    self.cnn2 = nn.Sequential(
      nn.Conv2d(16, 16, (3,3), padding=1),
      nn.ReLU())

    self.fc1 = nn.Sequential(
      nn.Linear(w*h*16, 64),
      nn.ReLU())
    self.fc2 = nn.Sequential(
      nn.Linear(64, 32),
      nn.ReLU())
    self.fc3 = nn.Linear(32, latent_dim)

  def forward(self, obs):
    cnn = self.cnn1(obs)
    cnn = self.cnn2(cnn)
    flat = cnn.flatten(start_dim=-3)
    return self.fc3(self.fc2(self.fc1(flat)))


class LatentToTrans(nn.Module):
  '''Predicts transition from Z and A.'''
  def __init__(self, latent_dim, action_dim=4):
    super().__init__()
    self.action_dim = action_dim
    self.fc1 = nn.Sequential(
      nn.Linear(latent_dim + action_dim, 64),
      nn.ReLU())
    self.fc2 = nn.Linear(64, latent_dim)

  def forward(self, obs, act):  # (*B,N), (*B,A) -> *B, N
    act_shape = act.shape
    act = to_one_hot(act.reshape(-1,), max_index=self.action_dim)
    obs = torch.cat([obs, act.reshape(*act_shape, self.action_dim)], dim=-1)
    return self.fc2(self.fc1(obs))

class LatentToReward(nn.Module):
  '''Predicts reward from latent space Z.'''
  def __init__(self, latent_dim):
    super().__init__()
    self.fc1 = nn.Sequential(
      nn.Linear(latent_dim, 64),
      nn.ReLU())
    self.fc2 = nn.Linear(64, 1)

  def forward(self, obs):
    return self.fc2(self.fc1(obs))


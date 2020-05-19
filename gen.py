import typing as t
import torch
import numpy as np

from torch.utils import data

from tqdm import trange
from env import KeyTask


def to_float(np_array):
  return np.array(np_array, dtype=np.float32)


class Transitions(data.Dataset):
  def __init__(self, experience_buffer: t.List[t.Mapping]):
    '''Construct a PyTorch dataset from replays'''
    self.experience_buffer = experience_buffer
    self.idx2episode = list()
    step = 0
    for ep in range(len(self.experience_buffer)):
      num_steps = len(self.experience_buffer[ep]['act'])
      idx_tuple = [(ep, idx) for idx in range(num_steps)]
      self.idx2episode.extend(idx_tuple)
      step += num_steps

    self.num_steps = step

  def __len__(self):
    return self.num_steps

  def __getitem__(self, idx):
    ep, step = self.idx2episode[idx]

    obs = to_float(self.experience_buffer[ep]['obs'][step])
    act = self.experience_buffer[ep]['act'][step]
    rew = to_float(self.experience_buffer[ep]['rew'][step])
    nobs = to_float(self.experience_buffer[ep]['nobs'][step])
    # NOTE(tk) likely will not be using goal
    #gobs = to_float(self.experience_buffer[ep]['goal'])

    return obs, act, rew, nobs


def gen_env(env: KeyTask, n_episodes=1000):
  '''Generate `n_episodes` times of (s, a, r, s').'''
  replay = []

  for episode in trange(n_episodes):
    obs = env.reset()
    replay.append({'obs':[], 'act':[], 'nobs': [], 'rew': [],
                    'goal': env.render_goal_state()})
    while True:
      replay[-1]['obs'].append(obs)
      assert env.valid_actions, 'No valid actions?!'
      action = np.random.randint(0, len(env.actions))
      obs, reward, done, _ = env.step(action)
      replay[-1]['rew'].append(reward)
      replay[-1]['act'].append(action)
      replay[-1]['nobs'].append(obs)
      if done: break

  return replay

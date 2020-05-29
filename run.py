
import click
import argparse
import logging
import pickle
import os
import sys

import numpy as np

import model
import gen

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch import optim

from env import KeyTask
from collections import Counter
from pathlib import Path
from tqdm import trange, tqdm


logging.basicConfig(
  format='[%(levelname)s:%(asctime)s] %(message)s',
  datefmt='%m%d@%H%M',
  handlers=[logging.FileHandler('log.txt','a'), logging.StreamHandler(sys.stdout)],
  level=logging.INFO)
logging.getLogger('skimage').setLevel(logging.CRITICAL)
logging.getLogger('model').setLevel(logging.DEBUG)

L = logging.getLogger(__name__)


class GroupExt(click.Group):
  def add_command(self, cmd, name=None):
    click.Group.add_command(self, cmd, name=name)
    for param in self.params:
      cmd.params.append(param)


to_device = lambda ctx, param, val: torch.device(val)


@click.command(cls=GroupExt)
@click.option('--episodes', default=512, type=int)
@click.option('--device', default='cuda', callback=to_device)
@click.option('--seed', default=42, type=int)
@click.option(
    '--task',
    default=lambda: os.environ.get('TASK', 'keytask'),
    type=click.Choice(['keytask']))
def common():
  pass


@click.option('--load', default='model_default.pt', type=click.Path(exists=True))
@click.option('--proto_samples', default=1024, type=int)
@common.command()
def plan(device, load, proto_samples, task, episodes, seed):
  Env = dict(keytask=KeyTask)[task]
  env = Env(seed=seed, max_steps=100)

  learner = model.Learner(
    device, latent_dim=50, in_shape=(3, 60, 60), action_dim=len(env.action_space))
  try:
    L.info('Loading state dict from %s', load)
    learner.load_state_dict(torch.load(load, map_location=device))
  except:
    L.warning('Load state dict failed. Trying to load entire object...')
    learner = torch.load(load, map_location=device)


  replay = gen.gen_env(env, n_episodes=episodes)
  tloader = gen.Transitions(replay)
  prototype_states = [
    tloader[idx][0] for idx in
    np.random.default_rng(seed).choice(np.arange(len(tloader)), proto_samples)
  ]

  learner.eval()
  planner = model.ValueIteration(learner, prototype_states)

  def predict(planner, env):
    planner.plan(goal_state=env.render_goal_state())

    done = False
    acs = []
    obs = env.render()

    while not done:
      action = planner.pi(obs, env.valid_actions).item()
      obs, reward, done, win = env.step(action)
      acs.append(action)

    return acs, win

  wins = 0
  act_counts = Counter()
  with trange(1, episodes+1) as t:
    for k in t:
      env.reset()
      acts, win = predict(planner, env)
      wins += win
      act_counts.update(acts)
      lens = sum(act_counts.values())
      t.set_postfix(wins=wins, acts=act_counts, L=len(acts), La=lens/k)
  L.info('Done evaluating planning. Wins: %s (avg %s)', wins, lens/k)


@click.option('--epochs', default=100, type=int)
@click.option('--save', default='model_default.pt', type=click.Path())
@click.option('--neg_samples', default=1, type=int)
@common.command()
def train(device, episodes, epochs, seed, save, neg_samples, task):

  Env = dict(keytask=KeyTask)[task]
  env = Env(seed=seed, max_steps=100)

  replay = gen.gen_env(env, n_episodes=episodes)
  tloader = gen.Transitions(replay)
  loader = data.DataLoader(tloader, batch_size=512, shuffle=True)

  learner = model.Learner(
    device, latent_dim=50, in_shape=(3, 60, 60), action_dim=len(env.action_space))

  optimizer = optim.Adam(learner.parameters(), lr=0.001)
  L.info('Starting training for %s epochs.', epochs)
  L.info('Best model will be saved to %s.', save)
  with trange(1, epochs+1) as t:
    best_loss = 1e9
    for epoch in t:
      loss_ = 0
      for step, batch in enumerate(tqdm(loader)):
        obs, action, reward, next_obs = batch = [x.to(device) for x in batch]
        optimizer.zero_grad()
        loss = learner.loss(*batch)
        loss.backward()
        loss_ += loss.item()
        optimizer.step()
      avg_loss = loss_ / len(loader.dataset)
      if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(learner.state_dict(), save)
      t.set_postfix(best=best_loss, avg=avg_loss)
  L.info('Done training. Best loss: %s', best_loss)


@click.command(cls=click.CommandCollection, sources=[common])
def cli():
  L.info('Called with args: %s', sys.argv)


if __name__ == '__main__':
  cli()


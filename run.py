import argparse
from pathlib import Path
import os
import sys
sys.path.append('..')

from tqdm import trange, tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch import optim
import numpy as np

import model, gen, env
from collections import Counter

import logging
import pickle

logging.basicConfig(
  format='[%(levelname)s:%(asctime)s] %(message)s',
  datefmt='%m%d@%H%M',
  handlers=[logging.FileHandler('log.txt','a'), logging.StreamHandler(sys.stdout)],
  level=logging.INFO)
logging.getLogger('skimage').setLevel(logging.CRITICAL)
logging.getLogger('model').setLevel(logging.DEBUG)

L = logging.getLogger(__name__)
print = L.info


def predict(planner, env):
  planner.plan(goal_state=env.render_goal_state())

  done = False
  acs = []
  obs = env.render()

  while not done:
    action = planner.pi(obs, env.valid_actions).item()
    # TODO? add new prototype states here maybe
    obs, reward, done, _ = env.step(action)
    acs.append(action)

  return acs, reward


def main(
    episodes=10, backup=10, epochs=1, seed=42, save=None, load=None,
    proto_samples=1024):
  keytask = env.KeyTask(seed=seed, max_steps=100)
  replay = gen.gen_env(keytask, n_episodes=episodes)
  tloader = gen.Transitions(replay)
  loader = data.DataLoader(tloader, batch_size=512, shuffle=True)

  device = torch.device('cuda')
  learner = model.Learner(device, latent_dim=50, in_shape=(3, 60, 60))
  if load:
    try:
      learner.load_state_dict(torch.load(load, map_location=device))
    except:
      L.warning('Load state dict failed. Trying to load entire object...')
      learner = torch.load(load, map_location=device)
  else:
    optimizer = optim.Adam(learner.parameters(), lr=0.001)
    L.info('Starting training for %s', epochs)
    with trange(1, epochs+1) as t:
      for epoch in t:
        loss_ = 0
        for step, batch in enumerate(tqdm(loader)):
          obs, action, reward, next_obs = batch = [x.to(device) for x in batch]
          optimizer.zero_grad()
          loss = learner.loss(*batch)
          loss.backward()
          loss_ += loss.item()
          optimizer.step()
        t.set_postfix(loss=loss_ / step)
    if save:
      torch.save(learner.state_dict(), save)
      L.info('Stored model to %s', save)

  with torch.no_grad():
    L.info('Using: keytask_train = env.KeyTask(seed=seed, max_steps=100)')
    keytask_train = env.KeyTask(seed=seed, max_steps=100)

    learner.eval()
    wins = 0
    act_counts = Counter()
    prototype_states = [
      tloader[idx][0] for idx in
      np.random.default_rng().choice(np.arange(len(tloader)), proto_samples)
    ]
    with trange(1, 1001) as t:
      for k in t:
        keytask_train.reset()
        planner = model.ValueIteration(learner, prototype_states, backup=backup)
        acts, reward = predict(planner, keytask_train)
        wins += (reward > 0)
        act_counts.update(acts)
        lens = sum(act_counts.values())
        t.set_postfix({
          'wins': wins, 'acts':act_counts, 'L': len(acts), 'La': lens / k })
    L.info('Done. Wins: %s (avg %s)', wins, lens/k)


def mkflags(include):
  parser = argparse.ArgumentParser()

  for k, v in include.items():
    if isinstance(v, tuple):
      v, type_ = v
    else:
      type_ = type(v)
    parser.add_argument(f'--{k}', default=v, type=type_)

  args_out = parser.parse_args()
  return args_out


if __name__ == '__main__':
  args = mkflags(dict(
    epochs=(100, int),
    episodes=(512, int),
    backup=(500, int),
    seed=(42, int),
    save=('model_default.pt', str),
    load=(None, str),
    proto_samples=(1024, int),
  ))

  L.info('Called with args: %s', args)
  # pickle.dump({'args': args}, open(meta_file, "wb"))

  try:
    main(**args.__dict__)
  except:
    L.exception('Unhandled exception in main.')

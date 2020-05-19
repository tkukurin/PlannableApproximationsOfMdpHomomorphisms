'''Simple key collection task implementation.
'''

import gym
import numpy as np
import typing
import collections
import skimage
from skimage import draw


class Obj:
  def __init__(self, name, pos=None):
    self.name=name
    self.pos=pos

class Key(Obj):
  def __init__(self, pos=None):
    super().__init__('key', pos)

class Player(Obj):
  def __init__(self, pos=None):
    super().__init__('player', pos)
    self.inv = set()
  def pickup(self, item, world):
    self.inv.add(item)
    world.remove(item)

def get_colors(cmap='Set1', num_colors=9):
  import matplotlib.pyplot as plt
  cm = plt.get_cmap(cmap)
  colors = []
  for i in range(num_colors):
    colors.append((cm(1. * i / num_colors)))
  return colors


class KeyTask(gym.Env):
  '''Positions are all (x,y). Collect key and deliver it to one corner.'''

  Ns_str = 'lrud'
  Ns = [(-1,0),(1,0),(0,-1),(0,1)]

  def __init__(self, seed=42, max_steps=100):
    self.max_steps = max_steps
    self.actions = self.Ns
    self.action_space = [0,1,2,3]
    self.width = 6
    self.height = 6

    self.seed(seed)
    self.reset()

  def place(self, obj, pos):
    # There are only 2 objects so no need to complicate with multiple items.
    if obj.pos in self.map:
      del self.map[obj.pos]
    self.map[pos] = obj
    obj.pos = pos

  def remove(self, obj):
    del self.map[obj.pos]
    self.objects.remove(obj)

  def _render(self, objects):
    im = np.zeros((self.height*10, self.width*10, 3), dtype=np.float32)
    for idx, obj in enumerate(objects):
      x, y = obj.pos
      rr, cc = skimage.draw.circle(
        y*10 + 5, x*10 + 5, 5, im.shape)
      im[rr, cc, :] = self.n2color[obj.name]
    return im.transpose([2, 0, 1])

  def render(self):
    return self._render(self.objects)

  def render_goal_state(self):
    '''player in goal state, no key present'''
    fake_player = Player(pos=self.goal_state)
    return self._render([fake_player])

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.curstep = 0
    self.player = Player()
    self.key = Key()
    self.map = {}

    self.objects = [self.player, self.key]

    randint = self.random.integers

    W, H = self.width, self.height
    corners = [(0,0), (0, H-1), (W-1, 0), (W-1,H-1)]
    self.goal_state = corners[randint(0, 4)]

    kpos = randint(0, W-1), randint(0, H-1)
    while kpos == self.goal_state:
      kpos = randint(0, W-1), randint(0, H-1)
    self.place(self.key, kpos)

    ppos = randint(0, W-1), randint(0, H-1)
    while ppos == self.goal_state or ppos == kpos:
      ppos = randint(0, W-1), randint(0, H-1)
    self.place(self.player, ppos)

    # NOTE in the paper they use 1 color per channel
    colors = list(map(lambda c: c[:3], get_colors(num_colors=9)))
    self.n2color = {
      self.player.name: colors[0],
      self.key.name: colors[1]
    }

    return self.render()

  def move(self, obj, delta):
    npos = (obj.pos[0] + delta[0], obj.pos[1] + delta[1])
    if self.is_valid(npos):
      self.place(obj, npos)

  def is_valid(self, position):
    x, y = position
    if not (0 <= x < self.width and 0 <= y < self.height):
      return False

    obj_in_position = self.map.get(position)
    if obj_in_position and obj_in_position != self.key:
      return False

    return True

  @property
  def valid_actions(self):
    x, y = self.player.pos
    return [
      i
      for i, (dx, dy) in enumerate(self.actions)
      if self.is_valid((x + dx, y + dy))
    ]

  def step(self, action):
    self.curstep += 1

    self.move(self.player, self.actions[action])

    reward = None
    if self.key in self.objects and self.player.pos == self.key.pos:
      self.player.pickup(self.key, self)
      reward = 1  # reward key pickup

    # NOTE(tk) Different dynamics than the paper.
    # Their reward is 0 if reaching goal without the key.
    done = self.player.pos == self.goal_state
    if done:
      reward = 1 if self.key in self.player.inv else -1
    elif not reward:
      reward = -.1

    won = reward == 1
    return self.render(), reward, done or self.curstep == self.max_steps, won


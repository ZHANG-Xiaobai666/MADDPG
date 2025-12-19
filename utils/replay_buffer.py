import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs):
        self.buffer.append((obs, action, reward, next_obs))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs = zip(*transitions)
        return obs, action, reward, next_obs

    def size(self):
        return len(self.buffer)
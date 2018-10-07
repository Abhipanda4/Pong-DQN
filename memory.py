import numpy as np
import random
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity, bs):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = bs

    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, self.batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

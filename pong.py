import torch
from torch.autograd import Variable
import os
from train import get_action, get_env
from model import DQN

env = get_env("PongNoFrameskip-v4", 4)

model_path = "dqn.pth"
dqn = DQN(env.observation_space.shape, env.action_space.n)
if os.path.exists(model_path):
    dqn.load_state_dict(torch.load(model_path))

S = env.reset()
done = False
while not done:
    S = Variable(torch.FloatTensor(S).unsqueeze(0))
    q_vals = dqn(S).max(1)
    A = q_vals[1].data[0].item()
    S_prime, R, done, _ = env.step(A)
    S = S_prime
    env.render()

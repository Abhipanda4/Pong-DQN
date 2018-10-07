import torch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from wrappers import wrap_deepmind, make_atari, wrap_pytorch
from model import *
from memory import *

def get_env(env_id, frame_stack):
    env = make_atari(env_id)
    env = wrap_deepmind(env, frame_stack)
    env = wrap_pytorch(env)
    return env

def get_epsilon(frame_idx, eps_start=1, eps_final=0.1, decay_win=30000):
    eps = eps_final + (eps_start - eps_final) * np.exp(-1 * frame_idx / decay_win)
    return eps

def get_action(model, env, state, epsilon):
    if np.random.uniform() > epsilon:
        with torch.no_grad():
            q_value = model(state)
            action = q_value.max(1)[1].data[0]
    else:
        action = np.random.randint(0, env.action_space.n)
    return action

def train_agent(args, device, verbose=True):
    '''
    Main driver function which trains the agent
    :param args: all args received from cmd
    :param device: whther to train on cpu or gpu
    :param verbose: whether to print info on console while training
    '''

    env = get_env(env_id=args.env_id, frame_stack=args.frame_stack)
    online_dqn = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_dqn = DQN(env.observation_space.shape, env.action_space.n).to(device)

    # target_dqn is initialized to same weights as online_dqn
    target_dqn.load_state_dict(online_dqn.state_dict())

    # adam optimizer for online dqn
    optimizer = optim.Adam(online_dqn.parameters(), lr=args.lr)

    # initialize replay memory
    buffer = ReplayBuffer(args.capacity, args.batch_size)

    # initialize number of episodes, rewards and loss
    ep = 0
    total_loss = 0
    total_reward = 0
    loss_list = []
    reward_list = []

    S = env.reset()
    for idx in range(1, args.num_frames_to_train + 1):
        eps = get_epsilon(idx)
        var_S = Variable(torch.FloatTensor(S).unsqueeze(0)).to(device)
        A = get_action(online_dqn, env, var_S, eps)
        S_prime, R, is_done, _ = env.step(A)

        buffer.push(S, A, R, S_prime, is_done)
        total_reward += R

        if is_done:
            S = env.reset()
            ep += 1
            if verbose:
                print("Episode: [%3d] complete - reward obtained: [%.2f]" %(ep, total_reward))
            loss_list.append(total_loss)
            total_loss = 0
            reward_list.append(total_reward)
            total_reward = 0
            torch.save(online_dqn.state_dict(), "dqn.pth")
        else:
            S = S_prime

        # perform weight updates by sampling from replay memory
        if len(buffer) > args.warm_up:
            batch_S, batch_A, batch_R, batch_Sp, batch_done = buffer.sample()

            # convert into variables
            batch_S = Variable(torch.FloatTensor(batch_S)).to(device)
            batch_Sp = Variable(torch.FloatTensor(batch_Sp)).to(device)
            batch_R = Variable(torch.FloatTensor(batch_R)).to(device)
            batch_done = Variable(torch.FloatTensor(batch_done)).to(device)
            batch_A = Variable(torch.LongTensor(batch_A)).to(device)

            with torch.no_grad():
                # find max[Q(S', A')] using target_dqn
                Q_Sp_A = target_dqn(batch_Sp).max(1)[0].squeeze()
                # target = R + gamma * max[Q'(S', A')]
                target = batch_R + args.gamma * Q_Sp_A * (1 - batch_done)

            y = online_dqn(batch_S).squeeze().gather(1, batch_A.unsqueeze(1)).squeeze()

            loss = (y - target).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if idx % args.update_target == 0:
            target_dqn.load_state_dict(online_dqn.state_dict())

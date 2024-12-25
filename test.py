import torch

from utils import *

from dqn import DQN
from relay_buffer import ReplayBuffer
from prepare_data import prepare_data
from enviroment import TradingEnvironment

num_episodes = 500
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 500
target_update = 10
buffer_capacity = 1e8

dataloader = prepare_data()
data = next(dataloader)

env = TradingEnvironment(data, initial_balance = 1e10)
env.reset()

state_dim, action_dim = env.observation_space.shape[0], env.action_space.n 

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-6)
replay_buffer = ReplayBuffer(buffer_capacity)



# 학습 루프
epsilon = epsilon_start
for episode in range(num_episodes):
    state = env.reset()
    total_reward = train(
        env = env,
        policy_net = policy_net, 
        target_net = target_net, 
        replay_buffer = replay_buffer, 
        optimizer = optimizer, 
        batch_size = batch_size, 
        gamma = gamma, 
        epsilon = epsilon, 
        data_len = len(data), 
        state = state, 
        action_dim = action_dim)

    # 타겟 네트워크 업데이트
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 탐색 비율 감소
    epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)

    print(f"Episode {episode}, Total Reward: {total_reward}")
import numpy as np
import torch

def select_action(state, policy_net, epsilon, action_dim):
    if np.random.rand() < epsilon:  # 무작위 행동
        return np.random.randint(action_dim), np.random.randint(1,100)
    else:  # 모델이 선택한 가장 좋은 행동
        with torch.no_grad():
            q_action, q_values = policy_net(state)
        return q_action.argmax().item(), q_values


def optimize_model(policy_net, target_net, replay_buffer, optimizer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return 0

    # 미니배치 샘플링
    transitions = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Q(s, a) 계산
    q_action = policy_net(states)
    q_action = q_action.gather(1, actions)

    # Q(s', a') 계산
    next_q_values = target_net(next_states).max(1)[0].detach()
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # 손실 계산
    loss = torch.nn.MSELoss()(q_action, target_q_values.unsqueeze(1))

    # 네트워크 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def train(env, policy_net, target_net, replay_buffer, optimizer, batch_size, gamma, epsilon, data_len, state, action_dim):
    total_reward = 0
    avg_loss, cnt = 0, 0
    for i in range(data_len):  # 각 에피소드의 최대 타임스텝
        action, value = select_action(state, policy_net, epsilon, action_dim)
        next_state, reward, done, _ = env.step(action, value)
        total_reward += reward

        replay_buffer.push((state, action, reward, next_state, done))
        state = next_state

        # 모델 업데이트
        loss = optimize_model(policy_net, target_net, replay_buffer, optimizer, batch_size, gamma)

        avg_loss += loss
        cnt += 1

        if i % 1000 == 0:
            env.render()
            print(f"Loss : {avg_loss / cnt}")
            

        if done:
            break
    return total_reward
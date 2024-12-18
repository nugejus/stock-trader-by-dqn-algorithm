import numpy as np
import gym
from gym import spaces

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000):
        """
        강화학습 트레이딩 환경 초기화
        Args:
            data (pd.DataFrame): 시계열 OHLCV 데이터
            initial_balance (float): 초기 계좌 잔고
        """
        super(TradingEnvironment, self).__init__()
        
        # 환경 변수 초기화
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.max_steps = len(data) - 1
        self.position = 0  # 보유 포지션 (0: 없음, 1: 매수, -1: 매도)

        # 관측 공간: OHLCV 데이터와 추가 상태 정보
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(data.columns) + 3,), dtype=np.float32
        )
        # 행동 공간: [0: 관망, 1: 매수, 2: 매도]
        self.action_space = spaces.Discrete(3)
        
    def reset(self):
        """
        환경 초기화 및 초기 상태 반환
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0
        return self._get_observation()
    
    def step(self, action):
        """
        한 스텝 진행
        Args:
            action (int): 에이전트의 행동 (0: 관망, 1: 매수, 2: 매도)
        Returns:
            observation (np.array): 관측 상태
            reward (float): 보상
            done (bool): 에피소드 종료 여부
            info (dict): 추가 정보
        """
        self._take_action(action)
        self.current_step += 1
        
        # 보상 계산 (순자산의 변화)
        reward = self.net_worth - self.initial_balance
        
        # 에피소드 종료 조건
        done = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """
        현재 상태 반환
        """
        obs = np.append(
            self.data.iloc[self.current_step].values,
            [self.balance, self.net_worth, self.position]
        )
        return obs
    
    def _take_action(self, action):
        """
        행동 수행
        """
        current_price = self.data.iloc[self.current_step]['close']
        
        if action == 1 and self.position == 0:  # 매수
            self.position = 1
            self.balance -= current_price
            self.net_worth = self.balance + current_price
        elif action == 2 and self.position == 1:  # 매도
            self.position = 0
            self.balance += current_price
            self.net_worth = self.balance
        
        # 관망일 경우 net_worth는 그대로 유지
        self.net_worth = self.balance + (self.position * current_price)

    def render(self, mode='human'):
        """
        환경 시각화 (옵션)
        """
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Net Worth: {self.net_worth}")
        print(f"Position: {self.position}")

# 데이터 예시 (OHLCV 포맷 필요)
import pandas as pd
data = pd.DataFrame({
    'open': [100, 102, 105],
    'high': [101, 104, 108],
    'low': [99, 100, 103],
    'close': [100, 103, 106],
    'volume': [1000, 1200, 1500]
})

# 환경 초기화
env = TradingEnvironment(data)
state = env.reset()

# 예제 스텝 실행
for _ in range(3):
    action = env.action_space.sample()  # 랜덤 행동
    state, reward, done, _ = env.step(action)
    env.render()
    if done:
        break

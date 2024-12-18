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
        self.btc = 0
        self.max_steps = len(data) - 1

        # 관측 공간: OHLCV 데이터와 추가 상태 정보
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(data.columns) + 3,), dtype=np.float32
        )
        # 행동 공간: [0: 관망, 1: 매수, 2: 매도]
        self.action_space = spaces.Discrete(3)
    
    def updata_data(self, new_data):
        self.data = new_data
    
    def reset(self):
        """
        환경 초기화 및 초기 상태 반환
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0
        self.btc = 0
        return self._get_observation()
    
    def step(self, action, value):
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
        self._take_action(action, value)
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
            [self.balance, self.btc, self.net_worth]
        )
        return obs
    
    def buy(self, current_price, value):
        if self.balance >= current_price * value:
            self.btc += value
            self.balance -= current_price * value
    
    def sell(self, current_price, value):
        if self.btc > 0:
            self.btc -= value
            self.balance += current_price * value
        

    def _take_action(self, action, value):
        """
        행동 수행
        param:
            action : 1 - 매수
                     2 - 매도
                     0 - 관망
            value - How many coins(float)
        """
        current_price = self.data.iloc[self.current_step]['close']
        
        if action == 1:  # 매수
            self.buy(current_price, value)
        elif action == 2:  # 매도
            self.sell(current_price, value)

        self.net_worth = self.balance + self.btc * current_price

    def render(self, mode='human'):
        """
        환경 시각화 (옵션)
        """
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance} USD")
        print(f"BTC: {self.balance} USD")
        print(f"Net Worth: {self.net_worth} USD")

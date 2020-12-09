import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler

import gym
from gym import spaces

class MultiStockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    """
    A 3-stock trading environment.
    State: vector of size 7 (n_stock * 2 + 1)
        - # shares of stock 1 owned
        - # shares of stock 2 owned
        - # shares of stock 3 owned
        - price of stock 1 (using daily close price)
        - price of stock 2
        - price of stock 3
        - cash owned (can be used to purchase more stocks)
    Action: categorical variable with 27 (3^3) possibilities
        - for each stock, you can:
        - 0 = sell
        - 1 = hold
        - 2 = buy
    """
    def __init__(self, data, n_stock, n_forecast, n_sentiment, initial_investment, agent):
        # data
        self.agent = agent
        self.stock_price_history = data
        self.n_stock = n_stock
        self.n_forecast = n_forecast
        self.n_sentiment = n_sentiment
        self.n_step, _ = self.stock_price_history.shape

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.stock_forecast = None
        self.stock_sentiment = None
        self.cash_in_hand = None
        
        if self.agent == "DQN":
            self.action_space = np.arange(3**self.n_stock)
        elif self.agent == "DDPG":
            self.action_space = spaces.Box(low = np.array([0]*self.n_stock), 
                                           high = np.array([self.n_stock-1]*self.n_stock),
                                           dtype=np.int8)

        # action permutations
        # returns a nested list with elements like:
        # [0,0,0]
        # [0,0,1]
        # [0,0,2]
        # [0,1,0]
        # [0,1,1]
        # etc.
        # 0 = sell
        # 1 = hold
        # 2 = buy
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

        # calculate size of state
        # # of stocks * 2 [# of stock stock_owned, stock_price]
        # + 1 [cash_in_hand]
        self.state_dim = self.n_stock * 2 + 1 + self.n_forecast + self.n_sentiment

        self.observation_space = spaces.Box(low=0, high=np.inf, 
                                            shape = (self.n_step, self.state_dim),
                                            dtype=np.float16)

        self.reset()


    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step][:self.n_stock]

        if self.n_forecast and self.n_sentiment:
            self.stock_forecast = self.stock_price_history[self.cur_step][self.n_stock:2*self.n_forecast]
            self.stock_sentiment = self.stock_price_history[self.cur_step][2*self.n_forecast:3*self.n_sentiment]
        elif self.n_forecast:
            self.stock_forecast = self.stock_price_history[self.cur_step][self.n_stock:2*self.n_forecast]
        if self.n_sentiment:
            self.stock_sentiment = self.stock_price_history[self.cur_step][self.n_stock:2*self.n_sentiment]

        self.cash_in_hand = self.initial_investment
        return self._get_obs()


    def step(self, action):
        #assert action in self.action_space

        # get current value before performing the action
        prev_val = self._get_val()

        # update price, i.e. go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step][:self.n_stock]

        # perform the trade
        self._trade(action)

        # get the new value after taking the action
        cur_val = self._get_val()

        # reward is the increase in porfolio value
        reward = cur_val - prev_val

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        # store the current value of the portfolio here
        info = {'cur_val': cur_val}

        # conform to the Gym API
        return self._get_obs(), reward, done, info


    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2*self.n_stock] = self.stock_price

        if self.n_forecast and self.n_sentiment:
            obs[2*self.n_stock:3*self.n_forecast] = self.stock_forecast
            obs[3*self.n_stock:4*self.n_sentiment] = self.stock_sentiment
        elif self.n_forecast:
            obs[2*self.n_stock:3*self.n_forecast] = self.stock_forecast
        elif self.n_sentiment:
            obs[2*self.n_stock:3*self.n_sentiment] = self.stock_sentiment

        obs[-1] = self.cash_in_hand
        return obs
        

    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand


    def _trade(self, action):
        # index the action we want to perform
        # 0 = sell
        # 1 = hold
        # 2 = buy
        # e.g. [2,1,0] means:
        # buy first stock
        # hold second stock
        # sell third stock
        
        if self.agent == "DQN":
            action_vec = self.action_list[action]
        elif self.agent == "DDPG":
            action_vec = action

        # determine which stocks to buy or sell
        sell_index = [] # stores index of stocks we want to sell
        buy_index = [] # stores index of stocks we want to buy
        
        for i, a in enumerate(action_vec):
            if round(a) == 0:
                sell_index.append(i)
            elif round(a) == 2:
                buy_index.append(i)

        # sell any stocks we want to sell
        # then buy any stocks we want to buy
        if sell_index:
            # Note: to simplify the problem, when we sell, we will sell ALL shares of that stock
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            # Note: when buying, we will loop through each stock we want to buy,
            #       and buy one share at a time until we run out of cash
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1 # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False

    def render(self, mode='human', close=False):
        pass


def play_one_episode(agent, env, scaler, batch_size):
    # note: after transforming states are already 1xD
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        agent.update_replay_memory(state, action, reward, next_state, done)
        agent.replay(batch_size)
        state = next_state

    return info['cur_val']


def get_scaler(env):
    """
    Return scikit-learn scaler object to scale (standardize) the states
    """

    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler
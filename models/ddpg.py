import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.common.vec_env import SubprocVecEnv

def DDPGAgent(multi_stock_env, num_episodes):
    models_folder = 'saved_models'
    rewards_folder = 'saved_rewards'

    env = DummyVecEnv([lambda: multi_stock_env])
    
    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    
    # Hyper parameters
    GAMMA = 0.99
    TAU = 0.001
    BATCH_SIZE = 16
    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_LEARNING_RATE = 0.001
    BUFFER_SIZE = 500

    print("\nRunning DDPG Agent...\n")
    model = DDPG(MlpPolicy, env, 
                gamma = GAMMA, tau = TAU, batch_size = BATCH_SIZE,
                actor_lr = ACTOR_LEARNING_RATE, critic_lr = CRITIC_LEARNING_RATE,
                buffer_size = BUFFER_SIZE, verbose=1, 
                param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=50000)
    model.save(f'{models_folder}/rl/ddpg.h5')

    del model
    
    model = DDPG.load(f'{models_folder}/rl/ddpg.h5')
    obs = env.reset()
    portfolio_value = []

    for e in range(num_episodes):
        action, _states = model.predict(obs)
        next_state, reward, done, info = env.step(action)
        print(f"episode: {e + 1}/{num_episodes}, episode end value: {info[0]['cur_val']:.2f}")
        portfolio_value.append(round(info[0]['cur_val'], 3))

    # save portfolio value for each episode
    np.save(f'{rewards_folder}/rl/ddpg.npy', portfolio_value)

    print("\nDDPG Agent run complete and saved!")

    a = np.load(f'./saved_rewards/rl/ddpg.npy')

    print(f"\nCumulative Portfolio Value Average reward: {a.mean():.2f}, Min: {a.min():.2f}, Max: {a.max():.2f}")
    plt.plot(a)
    plt.title("Portfolio Value Per Episode (DDPG)")
    plt.ylabel("Portfolio Value")
    plt.xlabel("Episodes")
    plt.show()
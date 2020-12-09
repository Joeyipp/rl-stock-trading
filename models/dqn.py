import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.9  # discount rate
        self.epsilon = 0.6  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = mlp(state_size, action_size)


    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


    def replay(self, batch_size=32):
        # first check if replay buffer contains enough data
        if self.memory.size < batch_size:
            return

        # sample a batch of data from the replay memory
        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['s2']
        done = minibatch['d']

        # Calculate the tentative target: Q(s',a)
        target = rewards + (1 - done) * self.gamma * np.amax(self.model.predict(next_states), axis=1)

        # With the Keras API, the target (usually) must have the same
        # shape as the predictions.
        # However, we only need to update the network for the actions
        # which were actually taken.
        # We can accomplish this by setting the target to be equal to
        # the prediction for all values.
        # Then, only change the targets for the actions taken.
        # Q(s,a)
        target_full = self.model.predict(states)
        target_full[np.arange(batch_size), actions] = target

        # Run one training step
        self.model.train_on_batch(states, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)


### The experience replay memory ###
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=16):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])


def mlp(input_dim, n_action, n_hidden_layers=2, hidden_dim=64):
    """ A multi-layer perceptron """

    # input layer
    i = Input(shape=(input_dim,))
    x = i

    # hidden layers
    for _ in range(n_hidden_layers):
        x = Dense(hidden_dim, activation='relu')(x)
    
    # final layer
    x = Dense(n_action)(x)

    # make the model
    model = Model(i, x)
    
    model.compile(loss='mse', optimizer='adam')
    #print((model.summary()))
    return model
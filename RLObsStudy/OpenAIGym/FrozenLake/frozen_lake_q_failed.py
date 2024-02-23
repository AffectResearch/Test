import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_Training=False, render=True):
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human" if render else None)

    if(is_Training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open("frozen_lake8x8.pk1","rb")
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha/learning rate
    discount_factor_g = 0.9 # gamma or discount factor

    epsilon = 1 # 1 = 100% random actions
    epsilon_decay_rate = 0.0001 # epsilon decay rate 1/0.0001 = 10000
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0] #states: 0 to 63, 0=topleft corner, 63 = bottom right corner
        terminated = False # True, when fall in hole or reached goal
        truncated = False # True when actions > 200

        while(not terminated and not truncated):
            if is_Training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left, 1=down, 2=right, 3=up
            else:
                action = np.argmax(q[state,:])

            new_state, reward,terminated,truncated,_ = env.step(action)

            if is_Training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state
    
    epsilon = max(epsilon - epsilon_decay_rate, 0)

    if(epsilon==0):
        learning_rate_a = 0.0001

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0,t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig("frozen_lake8x8.png")

    if is_Training:
        f = open("frozen_lake8x8.pk1","wb")
        pickle.dump(q,f)
        f.close()

if __name__ == "__main__":
    run(1000, is_Training=True, render=True) 
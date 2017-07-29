

import models.PolicyGradient as PG
import models.QLearner as QL

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf


env = gym.make('MountainCar-v0')


pltdat = {}
check_alphs = []
#for alph in [0.1,0.3,1.0,3.0,10.0]:
#    for c in [1,2,4,8,16]:
for alph in [1.0]:
    for c in [3]:
        check_alphs.append((alph,c))

for alph, c in check_alphs:
    random.seed(1)
    tf.set_random_seed(1)
    env.seed(2)

    agent = QL.QAgent(2,3,eps=0.1,alpha=alph,gam=0.99,C=c)

    rev = []
    
    for i_episode in range(500):
        observation = env.reset()
    
        agent.eps = max((1 - i_episode / 200.0) * 1.0, 0.1)
        #agent.eps = 0.1

        total_reward = 0
        t = 0
        while True:
            if i_episode % 50 == 0:
                env.render(mode="rgb_array")
    
            action = agent.make_action(observation)
            action = np.argmax(action)
    
            #print(action)
            observation, reward, done, info = env.step(action)
    
            agent.report_reward(reward)
            total_reward += reward
    
            if done:
                agent.end_episode()
                print("Episode {} finished after {} timesteps with {} reward".format(i_episode, t+1, total_reward))
                break
            t += 1
        rev.append(total_reward)

    pltdat[(alph,c)] = rev


for alph, c in check_alphs:
    rev = pltdat[(alph,c)]
    tmp = plt.plot(range(len(rev)), rev, label=str((alph,c)))
    plt.legend()

plt.show()


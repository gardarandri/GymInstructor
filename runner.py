

import models.PolicyGradient as PG
import models.QLearner as QL

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf



env = gym.make('CartPole-v0')

pltdat = {}
check_alphs = [0.0001,0.0003,0.001,0.003,0.01]

for alph in check_alphs:
    random.seed(1)
    tf.set_random_seed(1)

    agent = QL.QAgent(4,2,eps=0.5,alpha=alph,gam=0.99)

    rev = []
    
    for i_eposode in range(500):
        observation = env.reset()
    
        if i_eposode % 100 == 0:
            agent.eps *= 0.9
            agent.eps = max(agent.eps, 0.01)
    
        total_reward = 0
        for t in range(200):
            #if i_eposode % 100 == 0:
            #    env.render()
            #else:
            #    env.render(close=True)
    
            action = agent.make_action(observation)
            action = np.argmax(action)
    
            #print(action)
            observation, reward, done, info = env.step(action)
    
            agent.report_reward(reward)# if not done else -10)
            total_reward += reward
    
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        rev.append(total_reward)

    pltdat[alph] = rev

for alph in check_alphs:
    rev = pltdat[alph]
    tmp = plt.plot(range(len(rev)), rev, label=str(alph))
    plt.legend()

plt.show()


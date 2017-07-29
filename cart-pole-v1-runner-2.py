import models.PolicyGradient as PG
import models.QLearner2 as QL

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import math


env = gym.make('CartPole-v1')

def net_function(Q, in_size, out_size):
    all_layers = [in_size] + [8,16] + [out_size]

    return_vars = []

    for i in range(1,len(all_layers)):
        with tf.variable_scope("layer"+str(i)):
            W1 = tf.Variable(tf.random_normal([
                all_layers[i],
                all_layers[i-1]
                ], stddev=1.0/math.sqrt(all_layers[i-1]*all_layers[i])))
            Q = tf.matmul(W1, Q)
            b1 = tf.Variable(tf.random_normal([
                all_layers[i],
                1
                ], stddev=1.0/math.sqrt(all_layers[i])))
            Q = tf.add(Q, b1)
            if i != len(all_layers)-1:
                #Q = tf.nn.relu(Q)
                Q = tf.maximum(Q,0.05*Q)
                #Q = tf.nn.sigmoid(Q)
    
            return_vars.append(W1)
            return_vars.append(b1)

    return Q, return_vars


pltdat = {}
check_alphs = []
for alph in [1.0]:
    for c in [1]:
        check_alphs.append((alph,c))

for alph, c in check_alphs:
    random.seed(1)
    tf.set_random_seed(1)
    env.seed(2)

    agent = QL.QAgent(4,2,net_function,eps=0.1,alpha=alph,gam=0.99,C=c)

    rev = []
    
    for i_episode in range(501):
        observation = env.reset()
        #print(observation)
    
        agent.eps = max((1 - i_episode / 200.0) * 1.0, 0.01)
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
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                break
            t += 1
        rev.append(total_reward)

    pltdat[(alph,c)] = rev

    agent.output_summary()
    X1 = reduce(lambda x,y: x + [y[0]], agent.Q_history, [])
    X2 = reduce(lambda x,y: x + [y[1]], agent.Q_history, [])

    #plt.plot(np.arange(len(X1)), X1)
    #plt.plot(np.arange(len(X2)), X2)
    #plt.show()

for alph, c in check_alphs:
    rev = pltdat[(alph,c)]
    tmp = plt.plot(range(len(rev)), rev, label=str((alph,c)))
    plt.legend()

plt.show()


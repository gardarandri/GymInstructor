import models.QLearner2 as QL

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import math


env = gym.make('Breakout-v0')

def conv_layer(x, filter_size, stride, out_channels):
    in_channels = x.get_shape().as_list()[-1]

    F = tf.Variable(tf.random_normal([filter_size, filter_size, in_channels, out_channels], stddev=1.0/math.sqrt(filter_size*filter_size*in_channels*out_channels)))
    b = tf.Variable(tf.random_normal([1]))

    x = tf.nn.conv2d(x, F, [1,stride,stride,1], padding="VALID") + b
    x = tf.maximum(x,0.01*x)

    return x, [F,b]


def net_function(Q, in_shape, out_shape):
    return_vars = []

    net_layout = [(5,8),(5,8),(5,8)]

    for fl,ch in net_layout:
        Q, vl = conv_layer(Q, fl, 3, ch)
        return_vars += vl

    dim = Q.get_shape().as_list()
    Q = tf.reshape(Q, [-1, np.prod(dim[1:])])
    Q = tf.transpose(Q)

    W = tf.Variable(tf.random_normal([out_shape, np.prod(dim[1:])], stddev=1.0/math.sqrt(out_shape*np.prod(dim[1:]))))
    b = tf.Variable(tf.random_normal([out_shape, 1], stddev=1.0/math.sqrt(out_shape)))
    Q = tf.matmul(W,Q) + b

    return_vars += [W,b]

    return Q, return_vars


pltdat = {}
check_alphs = []
k = 4
for alph in [1.0]:
    for c in [1]:
        check_alphs.append((alph,c))

for alph, c in check_alphs:
    random.seed(1)
    tf.set_random_seed(1)
    env.seed(2)

    agent = QL.QAgent([210,160,3],4,net_function,alpha=alph,gam=0.99,C=c)

    rev = []
    
    i_episode = 0
    rec = None
    while True:
        i_episode += 1
        try:
            open("stopfile")
            break
        except IOError:
            None

        observation = env.reset()
    
        agent.eps = max((1 - i_episode / 1000.0) * 1.0, 0.1)

        to_render = True
        try:
            open("renderfile")
        except IOError:
            to_render = False
            env.render(close=True)

        to_record = i_episode % 2 == 0

        if to_record:
            rec = gym.monitoring.VideoRecorder(env,"./breakoutvid_ep"+str(i_episode)+".mp4")

        total_reward = 0
        t = 0
        while True:
    
            action = agent.make_action(observation)
            action = np.argmax(action)
    
            #print(action)
            summed_reward = 0
            for i in range(k):
                observation, reward, done, info = env.step(action)
                summed_reward += reward
                if to_render:
                    env.render(mode="human")
                if to_record:
                    rec.capture_frame()
    
            agent.report_reward(summed_reward)
            total_reward += summed_reward
    
            if done:
                agent.end_episode()
                print("Episode {} finished after {} timesteps with {} reward".format(i_episode, t+1, total_reward))
                break
            t += 1
        rev.append(total_reward)
        if to_record:
            rec.close()

    pltdat[(alph,c)] = rev

    #agent.output_summary()
    #X1 = reduce(lambda x,y: x + [y[0]], agent.Q_history, [])
    #X2 = reduce(lambda x,y: x + [y[1]], agent.Q_history, [])

    #plt.plot(np.arange(len(X1)), X1)
    #plt.plot(np.arange(len(X2)), X2)
    #plt.show()

for alph, c in check_alphs:
    rev = pltdat[(alph,c)]
    tmp = plt.plot(range(len(rev)), rev, label=str((alph,c)))
    plt.legend()

plt.show()


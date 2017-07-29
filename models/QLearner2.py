
import tensorflow as tf
import numpy as np
import random
import math
import collections


class QAgent:
    def __init__(self, state_space_size, action_space_size, gen_net, eps=0.01, alpha=0.0003, gam=1.0, lam=0.99, C=20):
        self.s_dim = state_space_size
        self.a_dim = action_space_size
        
        self.W = {}

        self.at_state = None
        self.at_action = None
        self.last_state = None
        self.last_action = None

        self.gam = gam
        self.lam = lam
        self.alpha = alpha
        self.eps = eps
        self.gradient_clip = 1e+10

        self.number_of_actions = 0

        self.batch_size = 32
        self.memory = collections.deque()
        self.memory_size = 1000000
        self.optimizer = tf.train.AdadeltaOptimizer(self.alpha)
        self.num_train_step = 0
        self.C = C

        self._init_neural_net(gen_net)
        self._add_summary()

        self.sess = self._start_session()

        self.Q_history = []


    def _lrelu(self, x):
        return tf.maximum(x,0.05*x)

    def _init_neural_net(self, gen_net):
        # Net
        with tf.variable_scope("input"):
            self.state_input = tf.placeholder(tf.float32, [None]+self.s_dim)

        self.Q = self.state_input
        self.Q_target = self.state_input

        self.Q, self.train_vars = gen_net(self.Q,self.s_dim,self.a_dim)
        self.Q_target, assign_vars = gen_net(self.Q_target,self.s_dim,self.a_dim)

        self.assign_op = [asv.assign(trv) for trv, asv in zip(self.train_vars, assign_vars)]

        with tf.variable_scope("training"):
            #Train
            self.target = tf.placeholder(tf.float32, [self.a_dim, None], name="target_input")
            self.loss = tf.reduce_mean((self.Q - self.target)**2,name="loss")
            self.train = self.optimizer.minimize(self.loss, name="step", var_list=self.train_vars)


    def _clip(self, t):
        return tf.maximum(tf.minimum(t, self.gradient_clip), -self.gradient_clip)

    def _start_session(self):
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)

        return sess

    def _add_to_memory(self, done=False):
        self.memory.append((self.last_state, self.last_action, self.last_reward, self.at_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()


    def _train_step(self, assign=False):
        if len(self.memory) > self.batch_size:
            batch_sample = random.sample(self.memory, self.batch_size)

            last_state = np.array([s[0] for s in batch_sample])
            last_action = [np.argmax(s[1]) for s in batch_sample]
            last_reward = [s[2] for s in batch_sample]
            at_state = np.array([s[3] for s in batch_sample])
            is_done = [s[4] for s in batch_sample]

            feed_dict = {}

            feed_dict[self.state_input] = last_state

            q_target_last = self.sess.run(self.Q_target, feed_dict={
                self.state_input : last_state
                })
            q_target_at = self.sess.run(self.Q_target, feed_dict={
                self.state_input : at_state
                })

            target_in = np.zeros((self.a_dim, self.batch_size))
            for i in range(self.batch_size):
                target_in[:,i] = q_target_last[:,i]
                max_q = max(q_target_at[:,i])
                target_in[last_action[i],i] = last_reward[i] + self.gam*(max_q if not is_done[i] else 0)

            # (Q(last_state,action_taken) - r + max_a Q_target(next_state, a)

            feed_dict[self.target] = target_in

            #print("W before update")
            #print(self.sess.run(self.train_vars[0]))
            self.sess.run([self.state_input, self.loss, self.train], feed_dict = feed_dict)
            #print("W after update")
            #print(self.sess.run(self.train_vars[0]))

            #if self.num_train_step % self.C == 0:
            if self.num_train_step % self.C == 0 and assign == True:
                self.sess.run(self.assign_op)

            if assign == True:
                self.num_train_step += 1



    def _normal_step(self):
        None

    def make_action(self, state):
        self.last_state = self.at_state
        self.last_action = self.at_action

        self.at_state = state

        Q_out = self.sess.run(self.Q, feed_dict={
            self.state_input : np.array(state).reshape([1]+self.s_dim)
            })

        Q_out = Q_out.reshape((self.a_dim))
        self.Q_history.append(Q_out)
        #print(Q_out)

        best_action_indx = np.argmax(Q_out)
        action = np.zeros((self.a_dim))

        if random.random() < self.eps:
            action[random.choice(range(0,self.a_dim))] = 1.0
        else:
            action[best_action_indx] = 1.0

        self.at_action = action

        if self.number_of_actions > 2:
            self._add_to_memory()
            self._train_step()

        self.number_of_actions += 1

        return action

    def report_reward(self, reward):
        self.last_reward = reward

    def end_episode(self):
        if self.number_of_actions > 2:
            self._add_to_memory(done=True)
            self._train_step(True)

        # TODO: is this all that has to be reset?
        self.number_of_actions = 0
        self.at_state = None
        self.at_action = None
        self.last_state = None
        self.last_action = None

    def _add_summary(self):
        self.all_summaries = tf.summary.merge_all()

    def output_summary(self, path=None):
        if path == None:
            path = "summaries/test"
        test_writer = tf.summary.FileWriter(path)
        if self.all_summaries != None:
            summary = self.sess.run(self.all_summaries)
            test_writer.add_summary(summary)
        
        test_writer.add_graph(self.sess.graph)








if __name__=="__main__":
    import matplotlib.pyplot as plt

    random.seed(0)
    tf.set_random_seed(0)

    agent = PolicyGradientAgent(2,3)
    num_episodes = 500

    y = []
    for k in range(num_episodes):
        state = np.zeros((2))
        total_reward = 0
        for i in range(100):
            state[0] += random.random()
            ac = agent.make_action(state)
            ac = np.argmax(ac)
            if ac == 0:
                state[1] += 1.0
            elif ac == 1:
                state[1] += -1.0
            else:
                state[1] += 0.0
            
            r = -abs(state[0] - state[1])
            agent.report_reward(r)
            total_reward += r

        print(total_reward)
        print(agent.sess.run(agent.W["L1"]))
        y.append(total_reward)

    plt.plot(range(num_episodes),y)
    plt.show()


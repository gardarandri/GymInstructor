
import tensorflow as tf
import numpy as np
import random
import math


class QAgent:
    def __init__(self, state_space_size, action_space_size, eps=0.01, alpha=0.3, gam=1.0, lam=0.5):
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

        self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)

        self._init_neural_net()
        self.sess = self._start_session()


    def _init_neural_net(self, hidden_layers = None):
        # TODO: Add scopes

        # Neural net
        with tf.variable_scope("input"):
            #self.action_input = tf.placeholder(tf.float32,(self.a_dim), name="action")
            self.state_input = tf.placeholder(tf.float32,(self.s_dim), name="state")

        #Q_in = tf.concat([self.action_input, self.state_input], 0)
        Q_in = tf.reshape(self.state_input, [self.s_dim,1])

        all_layers = None
        if hidden_layers == None:
            all_layers = [Q_in.get_shape().as_list()[0]] + [3] + [self.a_dim]
        else:
            all_layers = [Q_in.get_shape().as_list()[0]] + hidden_layers + [self.a_dim]

        Q = Q_in

        self.train_vars = []
        with tf.variable_scope("NeuralNet"):
            for i in range(1,len(all_layers)):
                with tf.variable_scope("L"+str(i)):
                    self.W["L"+str(i)] = tf.Variable(tf.random_normal((
                        all_layers[i],
                        all_layers[i-1]
                        ),stddev=1.0/math.sqrt(all_layers[i]*all_layers[i-1])), name="W") 
                    self.W["L"+str(i)+"-bias"] = tf.Variable(tf.random_normal((
                        all_layers[i],
                        1
                        ),stddev=1.0/math.sqrt(all_layers[i])), name="b")
                    Q = tf.matmul(self.W["L"+str(i)], Q) + self.W["L"+str(i)+"-bias"]
                    if i != len(all_layers)-1:
                        Q = tf.nn.relu(Q)
                        #Q = tf.maximum(Q,0.01*Q)
                        #Q = tf.nn.sigmoid(Q)
                    self.train_vars.append(self.W["L"+str(i)])

        # trainer
        with tf.variable_scope("train"):
            self.Q_target = tf.placeholder(tf.float32, [self.a_dim,1], name="Q_targ")

            self.loss = tf.reduce_sum((Q - (self.gam*self.Q_target))**2)
            self.train = self.optimizer.minimize(self.loss)
    
            #grads_and_vars = self.optimizer.compute_gradients((Q - (self.reward_input + self.max_Q_input))**2, var_list=self.train_vars)
    
            #self.train = self.optimizer.apply_gradients(grads_and_vars, name="step") # TODO: Add global step
    
        self.Q = Q

    def _clip(self, t):
        return tf.maximum(tf.minimum(t, self.gradient_clip), -self.gradient_clip)

    def _start_session(self):
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)

        return sess

    def _train_step(self, done=False):
        feed_dict = {}

        #q_values = self.sess.run(self.Q, feed_dict={
        #    self.state_input : self.at_state
        #    })


        #q_values[np.argmax(q_values)] += self.last_reward
        #if not done:
        #    q_values = self.last_reward * np.ones((self.a_dim,1))

        q_last = self.sess.run(self.Q, feed_dict={
            self.state_input : self.last_state
            })

        q_at = self.sess.run(self.Q, feed_dict={
            self.state_input : self.at_state
            })

        q_last[np.argmax(self.last_action)] = q_at[np.argmax(q_at)] + self.last_reward



        feed_dict[self.state_input] = self.last_state
        feed_dict[self.Q_target] = q_last

        #print("W before update")
        #print(self.sess.run(self.W["L1"]))
        self.sess.run(self.train, feed_dict = feed_dict)
        #print("W after update")
        #print(self.sess.run(self.W["L1"]))
        #print("e:")
        #print(self.e)
        #print("next_e:")
        #print(res[0:len(self.e)])
        #print("")


    def _normal_step(self):
        None

    def make_action(self, state):
        self.last_state = self.at_state
        self.last_action = self.at_action

        self.at_state = state

        Q_out = self.sess.run(self.Q, feed_dict={
            self.state_input : state
            })

        Q_out = Q_out.reshape((self.a_dim))

        best_action_indx = np.argmax(Q_out)
        action = np.zeros((self.a_dim))

        if random.random() < self.eps:
            action[random.choice(range(0,self.a_dim))] = 1.0
        else:
            action[best_action_indx] = 1.0

        self.at_action = action

        if self.last_action != None:
            self._train_step()

        return action

    def report_reward(self, reward):
        self.last_reward = reward

    def end_episode(self):
        if self.last_action != None:
            _train_step(done=True)

        # TODO: is this all that has to be reset?
        self.at_state = None
        self.at_action = None
        self.last_state = None
        self.last_action = None

        self.e = [np.zeros(x.get_shape().as_list()) for x in self.train_vars]






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


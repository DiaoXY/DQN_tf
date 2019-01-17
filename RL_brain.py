# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 19:38:36 2019
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
@author: lenovo
"""
import numpy as np 
import pandas as pd
import tensorflow as tf

np.random.seed(1)#每次生成的随机数也相同
tf.set_random_seed(1)

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increament=None,
            out_graph=False
            ):
        self.n_actions=n_actions
        self.n_features=n_features
        self.lr=learning_rate
        self.gamma=reward_decay#衰减参数0.9
        self.epsilon_max=e_greedy#0.9表示90%的时间选择Q值最大的动作，剩余时间随机选择动作
        self.replace_target_iter=replace_target_iter#更换目标网络参数的步数间隔
        self.memory_size=memory_size
        self.batch_size=batch_size
        self.epsilon_increament=e_greedy_increament
        self.epsilon=0 if e_greedy_increament is not None else self.epsilon_max
        
        self.learn_step_counter=0# total learning step
        
        # initialize zero memory [s, a, r, s_]
        self.memory=np.zeros((self.memory_size,self.n_features*2+2))#（s,a,r,s_）
        
        self._bulid_net()
        self.sess=tf.Session()
        if out_graph:
            tf.summary.FileWriter('logs/',self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())
        
        t_params=tf.get_collection('target_net_params')
        e_params=tf.get_collection('eval_net_params')
        self.repalce_target_iter=[tf.assign(t,e) for t,e in zip(t_params,e_params)]
        
        self.cost_his=[]   
        
    def _bulid_net(self):
        #------------build evaluate network-------------#
        self.s=tf.placeholder(tf.float32,[None,self.n_features],name='s')
        self.q_target=tf.placeholder(tf.float32,[None,self.n_actions],name='Q_target')
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names=['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1=10
            w_initializer=tf.random_normal_initializer(0,0.3)
            b_initializer=tf.constant_initializer(0.1)
                
            with tf.variable_scope('layer1'):
                w1=tf.get_variable('w1',[self.n_features,n_l1],initializer=w_initializer,collections=c_names)
                b1=tf.get_variable('b1',[1,n_l1],initializer=b_initializer,collections=c_names)
                l1=tf.nn.relu(tf.matmul(self.s,w1)+b1)
                
            with tf.variable_scope('layer2'):
                w2=tf.get_variable('w2',[n_l1,self.n_actions],initializer=w_initializer,collections=c_names)
                b2=tf.get_variable('b2',[1,self.n_actions],initializer=b_initializer,collections=c_names)
                self.q_eval=tf.matmul(l1,w2)+b2
                
            with tf.name_scope('loss'):
                self.loss=tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
            with tf.name_scope('train'):
                self._train_op=tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)           
            
        #--------------build target network--------------#
        self.s_=tf.placeholder(tf.float32,[None,self.n_features],name='s_')
        with tf.variable_scope('target_net'):
            c_names=['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            
            with tf.variable_scope('layer1'):
                w1=tf.get_variable('w1',[self.n_features,n_l1],initializer=w_initializer,collections=c_names)
                b1=tf.get_variable('b1',[1,n_l1],initializer=b_initializer,collections=c_names)
                l1=tf.nn.relu(tf.matmul(self.s_,w1)+b1)
                
            with tf.variable_scope('layer2'):
                w2=tf.get_variable('w2',[n_l1,self.n_actions],initializer=w_initializer,collections=c_names)
                b2=tf.get_variable('b2',[1,self.n_actions],initializer=b_initializer,collections=c_names)
                self.q_next=tf.matmul(l1,w2)+b2      
            
    def store_transition(self,s,a,r,s_):
        if not hasattr(self,'memory_counter'):#hasattr() 函数用于判断对象是否包含对应的属性。
            self.memory_counter=0
        transition=np.hstack((s,[a,r],s_))#列表进行 行连接
        
        index=self.memory_counter % self.memory_size
        self.memory[index,:]=transition
        
        self.memory_counter+=1    
            
    def choose_action(self,observation):
        observation=observation[np.newaxis,:]    #在np.newaxis位置增加一维
        
        if np.random.uniform()<self.epsilon:
            action_value=self.sess.run(self.q_eval,feed_dict={self.s:observation})
            action=np.argmax(action_value)
        else:
            action=np.random.randint(0,self.n_actions)    
        return action
    
    def replace_target_params(self):
        t_params=tf.get_collection('target_net_params')
        e_params=tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t,e) for t,e in zip(t_params,e_params)])
         
    def learn(self):
        #check target net replace params
        if self.learn_step_counter % self.replace_target_iter==0:
            self.sess.run(self.repalce_target_iter)
            print("target net params has replaced")
        if self.memory_counter >self.memory_size:
            sample_index=np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index=np.random.choice(self.memory_counter,size=self.batch_size)
        batch_memory=self.memory[sample_index,:]
        
        q_next,q_eval=self.sess.run(
                [self.q_next,self.q_eval],
                feed_dict={
                        self.s_:batch_memory[:,-self.n_features:],
                        self.s:batch_memory[:,:self.n_features]
                }
                )
        # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
        # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
        # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
        # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
        # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
        # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.

        # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
        # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
        # 具体在下面还有一个举例说明.
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1     
            
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
             
            



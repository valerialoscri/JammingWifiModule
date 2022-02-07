import collections
from pickle import FALSE
from pickletools import optimize
from random import random
from re import A
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from ns3gym import ns3env
import random

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import tensorflow.python.keras.backend as K

tf.compat.v1.disable_eager_execution()

class ActorCritic:

    def __init__(self,env,sess):
        
        self.env = env
        self.sess = sess

        self.learning_rate = 0.1
        self.epsilon = 0.40
        self.epsilon_decay = .995
        self.gamma = 0.99
        self.tau = .25
        self.memory = collections.deque(maxlen=2000)
        self.epoch = 0
        self.reward_final = 0
        self.threshold = 1500

        #create model actor 

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _,self.target_actor_model=self.create_actor_model()

        self.actor_critic_grad = tf.compat.v1.placeholder(tf.float32,[None,1])

        actor_model_weights = self.actor_model.trainable_weights
       

        self.actor_grads = tf.gradients(self.actor_model.output,actor_model_weights,-self.actor_critic_grad)
       
        grads = zip(self.actor_grads,actor_model_weights)
    
        self.optimize = tf.optimizers.Adam(self.learning_rate).apply_gradients(grads)

        #create model critic

        self.critic_state_input,self.critic_action_input,self.critic_model = self.create_critic_model()
        _,_,self.target_critic_model = self.create_critic_model()

        print(self.critic_model.output)
        print(self.critic_action_input)

        self.critic_grads = tf.gradients(self.critic_model.output,self.critic_action_input)
        print(self.critic_grads)

        #Init

        self.sess.run(tf.compat.v1.global_variables_initializer())
        

    def create_actor_model(self):
        state_input = Input(shape=(4,))
        h1=Dense(24,activation="relu")(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        action = Dense(4, activation="softmax")(h3)
        

        model = Model(state_input,action)
        adam = Adam(lr=0.001)
        model.compile(loss="mse",optimizer=adam)
        return state_input,model

    def create_critic_model(self):
        state_input = Input(shape=(4,))
        state_h1 = Dense(24)(state_input)
        state_h2 = Dense(48)(state_h1)
        
        action_input = Input(shape=(1,))
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2,action_h1])
        merged_h1 = Dense(24,activation="relu")(merged)
        output = Dense(1,activation="softmax")(merged_h1)
        
        model = tf.keras.Model([state_input,action_input],output)
        
        adam = Adam(lr=0.001)
        model.compile(loss="mse",optimizer=adam)
        return state_input,action_input,model

    def remember(self,cur_state,action,reward,new_state,done):
        self.memory.append([cur_state,action,reward,new_state,done])

    def train(self):
        batch_size = 300
        print(len(self.memory))
        if len(self.memory) < batch_size:
            return
        
        if(self.reward_final < self.threshold):
            samples = random.sample(self.memory,len(self.memory))
            self._train_critic(samples)
            self._train_actor(samples)


    def _train_critic(self,samples):
        print("train_critic")
        for sample in samples:
            cur_state,action,reward,new_state,done = sample
            print(sample)
            
            print("test")
            new_state= np.reshape(new_state,[1,4])
            target_action = self.target_actor_model.predict(new_state)
            print(new_state)
            print(target_action)
            target_action = np.argmax(target_action)
            print(target_action)
            target_action =[target_action]
            target_action= np.reshape(target_action,[1,1])
            print(self.target_critic_model.predict([new_state,target_action]))
            future_reward = self.target_critic_model.predict([new_state,target_action])[0][0]
            print(self.target_critic_model.predict([new_state,target_action]))
            print(future_reward)
            reward = reward + self.gamma * future_reward
            print(reward)
            print(reward)
            reward=[reward]
            #action =  tf.convert_to_tensor(action)
            action_test = np.reshape(action,[1,1])
            self.critic_model.fit([cur_state,action_test],reward)

    def _train_actor(self,samples):
        print("trai_ac")
  
        for sample in samples:
            
            cur_state,action,reward,new_state,_= sample
            print(sample)
            cur_state= np.reshape(cur_state,[1,4])
            predicted_action = self.actor_model.predict(cur_state)
            print(predicted_action)
            predicted_action = np.argmax(predicted_action)
            predicted_action =[predicted_action]
            predicted_action = np.reshape(predicted_action,[1,1])
              
            grads = self.sess.run(self.critic_grads,feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize,feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })


            
    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_model_weights)):
            actor_target_weights[i] = actor_model_weights[i] * self.tau + actor_target_weights[i] * (1 -self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):

        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i]=critic_model_weights[i] * self.tau + critic_target_weights[i] * (1 -self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def update_reward(self, reward):
        self.reward_final = self.reward_final + reward
        print(self.reward_final)

    def act(self,cur_state):
        #self.epsilon *= self.epsilon_decay
        self.epoch = self.epoch + 1
     

        if self.epoch < 300 or np.random.random() < self.epsilon:
            print(self.env.action_space)
            return self.env.action_space.sample()
        else:
            action = self.actor_model.predict(cur_state,verbose=1)
            action_choose = np.argmax(action)
            print("choose")
            print(action)
            print(action_choose)
            return action_choose

       


sess= tf.compat.v1.Session()
port = 5557
env = ns3env.Ns3Env(port = port,startSim=False)
actor_critic = ActorCritic(env,sess)

num_trial = 1000
trial_len = 500

cur_state = env.reset()
action = env.action_space.sample()




while True:

    env.render()
   
    cur_state= np.reshape(cur_state,[1,env.observation_space.shape[0]])
    print("test")
    print(cur_state)
    action = actor_critic.act(cur_state)
    print("action22")
    print(action)
    

    #action = np.random.choice(np.arange(len(action_probs[0])),p=action_probs[0])
    #print(action_probs)
    #action_taken = np.argmax(action)
    #print(action_taken)
    #if(isinstance(action,int)==False):
    # action=np.reshape(action[0],1)
    #action = action.reshape((1, env.action_space.shape[0]))
    print(action)
    new_state,reward,done,_ = env.step(action)


  

    #action = np.reshape(action,[1,12])
    action = [action]
    print(action)
    
    actor_critic.remember(cur_state,action,reward,new_state,done)
    actor_critic.train()
    actor_critic.update_target()
    actor_critic.update_reward(reward)

    
    cur_state = new_state


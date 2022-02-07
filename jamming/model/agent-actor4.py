import collections
from pickle import FALSE
from pickletools import optimize
from random import random
import re
from sys import intern
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


class ActorCritic(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.common = layers.Dense(24,activation="relu")
        self.actor = layers.Dense(12)
        self.critic = layers.Dense(1)
        self.eps =  np.finfo(np.float32).eps.item()
        self.loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.optimize = tf.keras.optimizers.Adam(learning_rate=0.01)

    def call(self,inputs: tf.Tensor) -> Tuple[tf.Tensor,tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

    def env_step(self,action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ]:

        state, reward, done,_ = env.step(action)
        return(np.array(state,np.float32),
               np.array(reward,np.int32),
               np.array(done,np.int32))


    def tf_env_step(self,action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step,[action],[tf.float32,tf.int32,tf.int32])

    def run_episode(self, initial_state: tf.Tensor, model: tf.keras.Model, max_step: int) -> Tuple[tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor]:

        action_probs = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size= True)
        rewards = tf.TensorArray(dtype=tf.int32,size=0,dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_step):

            state = tf.expand_dims(state,0)

            action_logits_t, value = model(state)
            print(action_logits_t)
            action_logits_t = tf.reshape(action_logits_t, [1, 12])

            action = tf.random.categorical(action_logits_t,1)[0,0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            values = values.write(t, tf.squeeze(value))

            action_probs = action_probs.write(t,action_probs_t[0,action])

            state, reward, done = self.tf_env_step(action)
            state = tf.reshape(state,[1,12])
            state.set_shape(initial_state_shape)
            print(state)

            rewards = rewards.write(t,reward)

            if tf.cast(done,tf.bool):
                break

            action_probs = action_probs.stack()
            values = values.stack()
            rewards = rewards.stack()

            return action_probs,values,rewards,state

    
    def get_expected(self, rewards:tf.Tensor,gamma:float, standart: bool = True) -> tf.Tensor:
        
        print("essais")
        print(rewards)
        n = tf.shape(rewards)[0]
        print(n)
        returns = tf.TensorArray(dtype=tf.float32, size=n)
        print("test")
        print(returns)

        rewards = tf.cast(rewards[::-1],dtype=tf.float32)
        print("reward get ")
        print(rewards)
        discounted_sum = tf.constant(0.0,dtype=tf.float32)
        discounted_sum_shape = discounted_sum.shape

        for i in tf.range(n):
            reward = rewards[i]
            print(reward)
            discounted_sum = reward + gamma 
            print("dicount")
            print(discounted_sum)
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i,discounted_sum)
        returns = returns.stack()[::-1]
        print(returns)

        if standart:
            returns = ((returns - tf.math.reduce_mean(returns)) / 
                        (tf.math.reduce_std(returns) + self.eps))
        
        return returns

    def compute_loss(self,action_probs:tf.Tensor, values: tf.Tensor, returns:tf.Tensor)-> tf.Tensor:

        advanatges = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advanatges)

        critic_loss = self.loss(values, returns)

        return actor_loss + critic_loss

    def train_step(self,initial_state: tf.Tensor, model: tf.keras.Model, gamma:float, max_step_per_episode: int)-> Tuple[tf.Tensor,tf.Tensor]:
        with tf.GradientTape() as tape:

            action_probs, values, rewards, new_state = self.run_episode(initial_state,model,max_step_per_episode)


            returns = self.get_expected(rewards, gamma,False)
            print(action_probs)
            print(values)
            print("after values")
            print(returns)
            action_probs, values, returns =[
                tf.expand_dims(x,1) for x in [action_probs, values, returns]
            ]

            loss = self.compute_loss(action_probs, values, returns)

        grads = tape.gradient(loss,model.trainable_variables)

        self.optimize.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward, new_state

    

sess= tf.compat.v1.Session()
port = 5557
env = ns3env.Ns3Env(port = port,startSim=False)
num_action = env.action_space.n
model = ActorCritic()

min_episodes_criterion = 100
max_episodes = 100
max_step_per_episode = 10
gamma = 0.70
reward_threshold = 1
running_reward = 0


episodes_rewards = collections.deque(maxlen=min_episodes_criterion)
state = env.reset()
state = np.reshape(state, [1, 12])
status = True
reward_execute = 0
while True: 
    while(status == True):
        with tqdm.trange(max_episodes) as t:
            for i in t:
                
                initial_state = tf.constant(state,dtype=tf.float32)
                episode_reward, new_state = model.train_step(
                    initial_state, model, gamma, max_step_per_episode)
                
                print(episode_reward)
                episode_reward = int(episode_reward)

                episodes_rewards.append(episode_reward)
                running_reward = statistics.mean(episodes_rewards)
                print(running_reward)

                state = new_state

                
                t.set_postfix(
                    episode_reward = episode_reward, running_reward = running_reward)
                
                if i % 10 ==0:
                    pass
                if running_reward > reward_threshold:
                    print("seuil")
                    status = False
                    running_reward = 0
                    break
    if(status == False): 
        for i in range(1,max_step_per_episode +1):
            state = tf.expand_dims(state,0)
            action_probs, _ = model(state)
            action = np.argmax(np.squeeze(action_probs))

            state,reward,done,_= env.step(action)
            print(reward)
            reward_execute = reward_execute + reward
            state = tf.constant(state,dtype=tf.float32)
            print(state)
            print(reward_execute)
            # retrain
            if(reward_execute < - 5):
                print("I am true")
                status = True
                reward_execute = 0
                state = np.reshape(state, [1, 12])
                break


from math import gamma
from pickletools import optimize
import matplotlib.style
from ns3gym import ns3env
import random
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.optimizers import Adam
from collections import deque
import tensorflow.python.keras.backend as K
import numpy as np




import tensorflow as tf
from tensorflow.python.keras.engine.input_layer import Input
tf.compat.v1.disable_eager_execution()
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class AgentActor: 
    def __init__(self,environment,sess):

        #Initailize attributes

        self.env = environment
        self.sess = sess
    
       

        #Initialize param actor-critic

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau = .125


        self.memory = deque(maxlen=2000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _,self.target_actor_model = self.create_actor_model()


        #self.actor_critic_grad = tf.compat.v1.placeholder(tf.float32,[None,12])

        actor_model_weights = self.actor_model.trainable_weights
        #self.actor_grads = tf.gradients(self.actor_model.output,actor_model_weights,-self.actor_critic_grad)
        #grads = zip(self.actor_grads,actor_model_weights)
        #self.optimize = tf.optimizers.Adam(self.learning_rate)

        self.critic_state_input, self.critic_model = self.create_critic_model()
        _,  self.target_critic_model = self.create_critic_model()
 

        #self.critic_grads = tf.gradients(self.critic_model.output_names)

        # Initialize for later gradient calculations

        #self.sess.run(tf.compat.v1.global_variables_initializer())


    def create_actor_model(self):
        state_input = Input(shape=(12,))
        h1 = Dense(128,activation='relu')(state_input)
        output = Dense(2,activation='softmax')(h1)

        model = Model(state_input,output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse",optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(12,))
        state_h1 = Dense(128,activation='relu')(state_input)
        output = Dense(1)(state_h1)
        model  = Model(state_input, output)
        
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model


    def remenber(self,cur_state,action,reward,new_state,done):
        self.memory.append([cur_state,action,reward,new_state,done])

    def _train_actor(self,samples):
        for sample in samples:
            cur_state,action,reward,new_state,_ = sample
            predicted_action = self.actor_model.predict(cur_state)
            
            """grads = self.sess.run(self.critic_grads,feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sees.run(self.optimize,feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })"""

    def _train_critic(self,samples):
        for sample in samples:
            cur_state,action,reward,new_state,done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state,target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state,action],reward,verbose=0)

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory,batch_size)
        self._train_critic(samples)
        self._train_actor(samples)
    
    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_model_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_model_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    
    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def act(self,cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor_model.predict(cur_state)



sess= K.get_session()
port = 5557
env = ns3env.Ns3Env(port = port,startSim=False)

actor_critic = AgentActor(env,sess)
num_trial = 10000
trial_len = 500
action_probs_history = []
critic_value_history = []
rewards_history = []
eps = np.finfo(np.float32).eps.item()
action = env.action_space.sample()
optimizer = tf.optimizers.Adam(learning_rate=0.01)
huber_loss = tf.losses.Huber()

num_inputs = 12
num_actions = 12
num_hidden = 248

inputs = Input(shape=(num_inputs,))
common = Dense(num_hidden, activation="relu")(inputs)
action = Dense(num_actions, activation="softmax")(common)
critic = Dense(1)(common)
episode_reward = 0
running_reward = 0

model = Model(inputs=inputs, outputs=[action, critic])
state = env.reset()
while True:
    print("eee")
    with tf.GradientTape() as tape:
        
        print(state)
        state = np.reshape(state, len(state))
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state,0)

        action_probs, critic_value = model(state)
        tf.print(action_probs)
        critic_value_history.append(critic_value[0,0])
        rand_choices=action_probs[np.randint(len(action_probs),(1,))]
        print(rand_choices)
        action = np.random.choice(action,12,p=action_probs)
        action_probs_history.append(tf.math.log(action_probs[0,action]))
        
    
        state, reward, done, _ = env.step(action)
        print(reward)
        rewards_history.append(reward)
        episode_reward = episode_reward +1
        
        running_reward = 0.05 * episode_reward + (1-0.05) * running_reward

        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + 0.95 * discounted_sum
            returns.insert(0,discounted_sum)
        
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns)+eps)
        returns = returns.tolist()

        history = zip(action_probs_history,critic_value_history,returns)
        actor_losses = []
        critic_losses = []

        for log_prob,value,ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)
            critic_losses.append(huber_loss(tf.expand_dims(value,0), tf.expand_dims(ret,0)))
        
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value,model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

        print("*******")
        template = "running reward: {:.2f}"
        print(template.format(running_reward))
    



    
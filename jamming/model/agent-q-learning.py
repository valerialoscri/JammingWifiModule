import gym 
import tensorflow as tf
import numpy as np 
import matplotlib as mlp
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import Conv2D
from tensorflow import keras
from ns3gym import ns3env
import random
from datetime import datetime
from collections import deque, Counter

#tf.disable_v2_behavior()

port = 5557
env = ns3env.Ns3Env(port=port,startSim=False)
env.reset()
env.render()

ob_space = env.observation_space
ac_space = env.action_space.n

def q_network(X,name_scope):

    initializer = tf.variance_scaling_initializer()

    with tf.variable_scope(name_scope) as scope:


        layer_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(8,8),strides=4,padding='SAME',kernel_initializer=initializer)(X)
        tf.summary.histogram('layer_1',layer_1)

        layer_2 = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=2,padding='SAME',kernel_initializer=initializer)(layer_1)
        tf.summary.histogram('layer_2',layer_2)

        layer_3 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='SAME',kernel_initializer=initializer)(layer_2)
        tf.summary.histogram('layer_3',layer_3)

        flat = tf.keras.layers.Flatten()(layer_3)

        fc= tf.keras.layers.Dense(units=128,kernel_initializer=initializer)(flat)
        tf.summary.histogram('fc',fc)

        output = tf.keras.layers.Dense(units=ac_space,activation=None,kernel_initializer=initializer)(fc)
        tf.summary.histogram('output',output)

        vars ={v.name[len(scope.name):]: v for v in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}

        return vars, output


epsilon = 0.5
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 50000

def epsilon_greedy(action,step):
    p=np.random.random(1).squeeze()
    epsilon = max(eps_min,eps_max -(eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(ac_space)
    else:
        return action

buffer_len = 20000
exp_buffer= deque(maxlen=buffer_len)

def sample_memories(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem= np.array(exp_buffer)[perm_batch]
    return  mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4]

batch_size =48
input_shape =(None,88,80,1)
learning_rate =0.001
X_shape = (None, 88,80,1)

discount_factor = 0.97

global_step =0
copy_steps=100
steps_train =4
start_step=2000


logdir ='logs'

#tf.reset_default_graph()

# Now we define the placeholder for our input i.e game state
X = tf.placeholder(tf.float32, shape=X_shape)

# we define a boolean called in_training_model to toggle the training
in_training_mode = tf.placeholder(tf.bool)


mainQ, mainQ_outputs = q_network(X,'mainQ')

targetQ, targetQ_outputs = q_network(X, 'targetQ')

X_action = tf.placeholder(tf.int32, shape=(None,))
Q_action = tf.reduce_sum(targetQ_outputs * tf.one_hot(X_action, ac_space), axis=-1, keepdims=True)
print(mainQ.items())
#copy_op = [tf.assign(main_name, targetQ[var_name]) for var_name, main_name in mainQ.items()]
#copy_target_to_main = tf.group(*copy_op)


y = tf.placeholder(tf.float32, shape=(None,1))

# now we calculate the loss which is the difference between actual value and predicted value
loss = tf.reduce_mean(tf.square(y - Q_action))

# we use adam optimizer for minimizing the loss
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

loss_summary = tf.summary.scalar('LOSS', loss)
merge_summary = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())



with tf.Session() as sess:
    init.run()
    
    try:
        while True:
            done = False
            epoch = 0
            episodic_reward = 0
            actions_counter = Counter() 
            episodic_loss = []

            while not done:

                actions= mainQ_outputs.eval(feed_dict={X:[obs], in_training_mode:False})

                #get the action 

                action =np.argmax(actions,axis=-1)
                actions_counter[str(action)]+=1

                action - epsilon_greedy(action,global_step)

                next_obs,reward, done, _ = env.step(action)

                exp_buffer.append([obs,action,next_obs,reward,done])

                if global_step % steps_train ==0 and global_step >strat_step:

                    o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)

                    # states
                    o_obs = [x for x in o_obs]

                    # next states
                    o_next_obs = [x for x in o_next_obs]

                    # next actions
                    next_act = mainQ_outputs.eval(feed_dict={X:o_next_obs, in_training_mode:False})

                    y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1-o_done) 

                    # merge all summaries and write to the file
                    mrg_summary = merge_summary.eval(feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:False})
                    file_writer.add_summary(mrg_summary, global_step)

                    # now we train the network and calculate loss
                    train_loss, _ = sess.run([loss, training_op], feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:True})
                    episodic_loss.append(train_loss)
                
                # after some interval we copy our main Q network weights to target Q network
                #if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                   # copy_target_to_main.run()

                obs = next_obs
                epoch += 1
                global_step += 1
                episodic_reward += reward
            
            print('Epoch', epoch, 'Reward', episodic_reward,)


    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")
    finally:
        
        print ('Optimal Done')  
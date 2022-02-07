import gym 
import tensorflow as tf
import numpy as np 
import matplotlib as mlp
import matplotlib.pyplot as plt 
from tensorflow import keras
from ns3gym import ns3env

port = 5556
env = ns3env.Ns3Env(port=port,startSim=False)
env.reset()
env.render()

ob_space = env.observation_space
ac_space = env.action_space

print("Observation space: ",ob_space,ob_space.dtype)
print("Action space: ", ac_space,ac_space.n)


s_size = ob_space.shape[0]
a_size = ac_space.n
model = keras.Sequential()
model.add(keras.layers.Dense(s_size,input_shape=(s_size,),activation='relu'))
model.add(keras.layers.Dense(a_size,activation='softmax'))
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

total_episodes =200
max_env_steps = 100
env.max_episode_step = max_env_steps


epsilon = 0.5
epsilon_min = 0.01
epsilon_decay = 0.999

time_history=[]
rew_history = []
stepIdx=0
rewardsum = 0

try:

    state = env.reset()
    state=np.reshape(state,[1,s_size])
  

    while True:
        stepIdx +=1

        #choose action 

        if np.random.rand(1) <  epsilon:
            action = np.random.randint(a_size)
        else:
            action = np.argmax(model.predict(state)[0])

        #Step 

        next_state, reward, done,_ = env.step(action)

        if done: 
            print("episode : {}/{},time: {},rew: {}, eps: {:.2}".format(1,total_episodes,stepIdx,rewardsum,epsilon))
            break

        next_state = np.reshape(next_state,[1,s_size])

        #train
        target = reward
        if not done: 
            target = (reward +0.95 * np.amax(model.predict(next_state)[0]))
        
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f,epochs=1,verbose=0)

        state = next_state
        rewardsum +=reward

        if epsilon > epsilon_min: epsilon *= epsilon_decay

        time_history.append(stepIdx)
        rew_history.append(action)
        print("test")


except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    
    print(rew_history)
    print("Done")
    mlp.rcdefaults()
    mlp.rcParams.update({'font.size': 16})
    plt.grid(True,linestyle='--')
    plt.title('Learning Performance')
    #plt.plot(range(len(time_history)), time_history,label='Step',marker="^",linestyle=":")
    plt.plot(rew_history,label="Channel Selection",marker="",linestyle="-")
    plt.xlabel('Episode')
    plt.ylabel('Channel')
    plt.legend(prop={'size':12})

    plt.savefig('learning.pdf',bbox_inches='tight')
    print(rewardsum)
    plt.show()
    env.close()



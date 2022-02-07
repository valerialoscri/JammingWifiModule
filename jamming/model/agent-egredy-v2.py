import gym 
import tensorflow as tf
import numpy as np 
import matplotlib as mlp
import matplotlib.pyplot as plt 
from tensorflow import keras
from ns3gym import ns3env

def epsilon_greedy(epsilon):

    rand = np.random.random()
    if rand < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)

    return action

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
print(s_size)

num_rounds = 200000
count = np.zeros(10)
sum_rewards = np.zeros(10)
Q= np.zeros(10)
stepIdx = 0
time_history =[]
rew_history=[]
state = env.reset()
state=np.reshape(state,[1,s_size])
rewardsum = 0


try:    

   
    while True: 

        arm = epsilon_greedy(0.95)

        #get the reward 
        observation,reward, done, info = env.step(arm)
        print(reward)

        count[arm] +=1

        sum_rewards[arm]+=reward

        Q[arm] = sum_rewards[arm]/count[arm]
        
        stepIdx +=1

        time_history.append(stepIdx)
        rew_history.append(arm)
        rewardsum = rewardsum + reward
       

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
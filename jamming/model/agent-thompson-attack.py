import gym 
import tensorflow as tf
import numpy as np 
import matplotlib as mlp
import matplotlib.pyplot as plt 
from tensorflow import keras
from ns3gym import ns3env


def thompson_sampling(alpha,beta):
    samples = [np.random.beta(alpha[i]+1,beta[i]+1) for i in range(12)]

    return np.argmax(samples)

port = 5557
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
num_rounds = 12
rewarsum = 0

count = np.zeros(12)

sum_rewards = np.zeros(12)

Q = np.zeros(12)

alpha = np.ones(12)
beta = np.ones(12)
rew_history=[]
rew_history2=[]

try: 

    while True:
        
        #select the arms
        arm = thompson_sampling(alpha,beta)

        #get the reward 
        observation,reward, done, info = env.step(arm)

        if done: 
            print("episode : {}/{},time: {},rew: {}, eps: {:.2}".format(1,total_episodes,stepIdx,rewardsum,epsilon))
            break

        #update the count of the arm 

        count[arm] +=1

        #sum the rewards obtained from the arm 

        sum_rewards[arm] +=reward

        #calculate Q value which is the average rewards of the arm
        Q[arm] = sum_rewards[arm]/count[arm]

        #If it is a positive reward increment alpha
        if reward >0:
            alpha[arm] +=1
        
        else:
            beta[arm]+=11
        
        rew_history.append(arm)
        rewarsum = rewarsum + reward
        rew_history2.append(reward)

        
except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    
    print(rew_history)
    print("Done")
    print(rewarsum)
    print(rew_history2)
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
    plt.show()
    env.close()
print ('Optimal arm is {}.format(np.argmax(Q)))')
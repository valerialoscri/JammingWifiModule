import argparse
from ns3gym import ns3env

port = 5555
env = ns3env.Ns3Env(port=port,startSim=False)
env.reset()


ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ",ob_space,ob_space.dtype)
print("Action space",ac_space,ac_space.dtype)

stepIdx=0
currIt=0
iterationNum =3

try:

    obs= env.reset()
    print("Step: ",stepIdx)
    print("---obs: ",obs)

    while True:
        stepIdx +=1
        action = env.action_space.sample()
        print("---action: ",action)

        print("Step: ",stepIdx)
        obs, reward, done , info = env.step(action)
        print("--obs,reward,done,info:",obs,reward,done,info)

        if done:
            break

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    env.close()
    print("Done")



import gym
import gym_duckietown
import numpy
from DQAgent import DQNAgent
from PreprocessFrame import PreprocessFrame
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
no_sessions=0
no_frames=100

env=gym.make("Duckietown-straight_road-v0")
agent=DQNAgent(14400,2)
agent.load(dir_path+"\dqagent-ddqn.h5")
batch_size=32
for _ in range(no_sessions):
	env.seed(2)
	obs=env.reset()
	obs=PreprocessFrame(obs)
	for ep in range(no_frames):
		action=agent.act(obs)
		next_obs,rew,done,info=env.step(action[0])
		next_obs=PreprocessFrame(next_obs)
		agent.remember(obs, action[0], rew, next_obs, done)
		obs=next_obs
		env.render()
		print(rew)
		if len(agent.memory) > batch_size:
			agent.replay(batch_size)
		if ep%10 ==0:
			agent.save(dir_path+"\dqagent-ddqn.h5")
		if done:
			agent.update_target_model()
			agent.save(dir_path+"\dqagent-ddqn.h5")
			break
env.seed(2)
obs=env.reset()
for ep in range(no_frames):
		action=agent.act(obs)
		next_obs,rew,done,info=env.step(action[0])
		next_obs=PreprocessFrame(next_obs)
		obs=next_obs
		env.render()
		print(rew)
		if done:
			break
env.close()
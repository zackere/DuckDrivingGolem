import gym
import gym_duckietown
import numpy
from DQAgent import DQNAgent
from PreprocessFrame import PreprocessFrame
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
no_sessions=0
no_frames=500

env=gym.make("Duckietown-straight_road-v0")
env=gym_duckietown.wrappers.DiscreteWrapper(env)
agent=DQNAgent(120*120,3)
agent.load(dir_path+"\dqagent-ddqn.h5")
batch_size=32
for _ in range(no_sessions):
	env.seed(2)
	obs=PreprocessFrame(env.reset())
	for ep in range(no_frames):
		action=agent.act(obs)
		next_obs,rew,done,info=env.step(action)
		rew+=1000
		next_obs=PreprocessFrame(next_obs)
		agent.remember(obs, action, rew, next_obs, done)
		obs=next_obs
		#env.render()
		print(rew)
		if len(agent.memory) > batch_size:
			agent.replay(batch_size)
		if ep%10 ==0:
			agent.save(dir_path+'\dqagent-ddqn.h5')
		if done:
			agent.save(dir_path+'\dqagent-ddqn.h5')
			break
env.seed(2)
obs=PreprocessFrame(env.reset())
for ep in range(no_frames):
	action=numpy.argmax(agent.model.predict(obs)[0])
	obs,rew,done,info=env.step(action)
	obs=PreprocessFrame(obs)
	env.render()
	if done:
		break
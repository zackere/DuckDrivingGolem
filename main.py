import gym
import gym_duckietown
import numpy
from DQAgent import DQNAgent
from PreprocessFrame import PreprocessFrame
no_sessions=1000
no_frames=100

env=gym.make("Duckietown-straight_road-v0")
agent=DQNAgent(14400,2)
batch_size=32
for ep in range(no_sessions):
	env.seed(2)
	obs=env.reset()
	obs=PreprocessFrame(obs)
	for _ in range(no_frames):
		action=agent.act(obs)
		next_obs,rew,done,info=env.step(action[0])
		next_obs=PreprocessFrame(next_obs)
		agent.remember(obs, action[0], rew, next_obs, done)
		obs=next_obs
		env.render()
		if len(agent.memory) > batch_size:
			agent.replay(batch_size)
		if done:
			break
env.close()
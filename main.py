import gym
from EnviromentWithHistory import EnvWithHistoryWrapper
import gym_duckietown

no_sessions=1;
no_frames=1000;

env=gym.make("Duckietown-straight_road-v0")
env=EnvWithHistoryWrapper(env,range(5))
env.reset()

for _ in range(no_sessions):
	env.reset()
	for _ in range(no_frames):
		obs,rew,done,info=env.step([1,1]);
		env.render()
		if done:
			break
env.close()
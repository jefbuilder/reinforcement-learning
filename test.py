import gym
from network import agent
import numpy as np
import pickle
import os
env = gym.make('LunarLander-v2')
if os.path.exists("data.pickle"):
  
  with open("data.pickle", 'rb') as f:
    
    bot = pickle.load(f)
else:
  print("create a bot")
  bot = agent(gamma=0.99,epsilon=1.0,batch_size=64, n_actions=4,
            eps_end=0.01, input_dims=[8],lr=0.003)
scores, eps_history = [], []

n_games = 500 - int(input("how many times have you got : "))

for i in range(n_games):
  score = 0
  print('epoch ' + str(i))
  done = False
  observation = env.reset()
  
  while not done:
    action = bot.choose_action(observation)
    observation_, reward, done, info = env.step(action)
    score += reward
    bot.store_transition(observation, action, reward,
                          observation_, done)
    bot.learn()
    env.render()
    observation = observation_
    
      
  f = open("data.pickle", 'wb')
  pickle.dump(bot, f)
  f.close
  scores.append(score)
  eps_history.append(bot.epsilon)

  
  print("score : " + str(score))
  print("eps : " + str(bot.epsilon))

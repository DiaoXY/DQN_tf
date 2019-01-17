# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 19:39:26 2019


@author: lenovo
"""

from maze_env import Maze
from RL_brain import DeepQNetwork



def run_maze():
    step=0
    for episode in range(3000):
        observation=env.reset()
        
        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action=RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_,reward,done=env.step(action)
            
            RL.store_transition(observation,action,reward,observation_)
            
            if(step>200)and (step%5==0):
                RL.learn()
            
            observation=observation_
            # break while loop when end of this episode
            if done:
                break
            step += 1
    #end of game
    print("game over")
    env.destroy()
if __name__=="__main__":
    
    env=Maze()
    RL=DeepQNetwork(env.n_actions,env.n_features,
                    learning_rate=0.01,
                    reward_decay=0.9,
                    e_greedy=0.9,
                    replace_target_iter=200,
                    memory_size=200,       
            )
    env.after(100,run_maze())
    env.mainloop()
    RL.plot_cost()
           
            
            
            
            
            
            
            
            
            
            
            
        
        
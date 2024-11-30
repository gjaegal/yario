import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym

from PyQt5.QtWidgets import QApplication
import sys
from PPO import PPO
from Game import Game


#################################### Testing ###################################
def test(load_path):
    print("============================================================================================")

    ################## hyperparameters ##################

    env_name = "SuperMarioBros"
    has_continuous_action_space = False
    max_ep_len = 10000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 20               # update policy for K epochs
    eps_clip = 0.3              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = Game(x_pixel_num = 256, y_pixel_num = 240, visualize = True)

    # state space dimension
    state_dim = 16

    # action space dimension
    action_dim = 12

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    

    ppo_agent.load(load_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    time_step = 0
    prev_action = 0
    state = None

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(0, max_ep_len+1):
            
            if state==None:
                state, reward, done, _ = env.step_new(prev_action)
                ep_reward += reward
                continue

            # select action with policy
            time_step += 1
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step_new(action)
            prev_action = action
            ep_reward += reward

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))

    # env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    load_path = "ppo_agent_ver4_episode1199_604000.pth"
    
    test(load_path)
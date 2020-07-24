import os
import time
from copy import deepcopy
import copy
import math
import json
import random
import argparse
import subprocess 
import os
from datetime import datetime
import itertools
import pprint
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
matplotlib.rcParams["axes.linewidth"] = 1.1
from control.CEM import * 
from control.MPPI import * 
from control.RandomShooting import * 
from control.MultimodalCEM import * 
from agent import Agent
from logger import * 
from env import *
from envs.piche_env import *
from envs.planar_env import *
from measures import *

def save_reward(rewards,logdir, savedir):
    np.save(logdir + "/rewards.npy",np.array(rewards))
    subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
    print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
    now = datetime.now()
    current_time = str(now.strftime("%H:%M:%S"))
    subprocess.call(['echo','saved at time: ' + str(current_time)])
    print("Rewards saved")

def boolcheck(x):
    return str(x).lower() in ["true", "1", "yes"]

def test_mpc(env, mpc_agent, logger,num_epochs,logdir, savedir,plot_statespace = False):
    mpc_agent.set_action_noise(None)
    reward_list = []
    subprocess.call(['echo','Beginning test'])
    for n in range(num_epochs):
        subprocess.call(['echo','Epoch : ' + str(n)])
        states = []
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = mpc_agent(state)
            state, reward, done, _ = env.step(action)
            states.append(state)
            total_reward += reward
        logger.log(f"Reward: {total_reward}")
        reward_list.append(total_reward)
        #save each epoch
        save_reward(reward_list, logdir,savedir)
        if plot_statespace == True:
            states = np.stack(states)
            increment = 1/states.shape[0]
            alpha = 0.0
            for i in range(states.shape[0]):
                plt.scatter(states[i, 0], states[i, 1], color="b", alpha=min(1, alpha))
                alpha += increment
            try: 
                env._env.draw_env()
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                ax = plt.gca()
                plt.show()
            except:
                pass

                


def main(args):
    seed_all(args.seed)

    logger = Logger(args.logdir, args.seed)
    logger.log(f"\nStarting Planner Experiment [device: {args.device}]\n")
    logger.log(args)

    env = Env(
        args.env_name,
        max_episode_steps=args.max_episode_steps,
        action_repeat=args.action_repeat,
        seed=args.seed,
        device=args.device,
    )

    reward_measure = RewardMeasure(env)
    if args.planner == "cem":
        mpc_agent = CEMAgent(
            env,
            env.action_space.shape[0],
            args.ensemble_size,
            args.device,
            plan_horizon=args.plan_horizon,
            optimisation_iters=args.optimisation_iters,
            num_candidates=args.num_candidates,
            top_candidates=args.top_candidates,
            cem_alpha=args.cem_alpha,
            reward_measure=reward_measure,
        )
    #Random shooting planner
    elif args.planner == "shooting":
        mpc_agent = RandomShootingPlanner(env,args.ensemble_size,env.action_space.shape[0],args.plan_horizon,args.num_candidates,args.action_std,reward_measure,None,args.discount_factor,args.device)
    #Path Integral Planner
    elif args.planner == "mppi":
        mpc_agent = PIPlanner(env, env.action_space.shape[0],args.ensemble_size,args.num_candidates,args.plan_horizon,args.lambda_,args.noise_mu, args.noise_sigma,reward_measure,device=args.device)
    elif args.planner == "multimodal_cem":
        mpc_agent = MultiModalCEMPlanner(env,args.ensemble_size,env.action_space.shape[0],args.plan_horizon,args.num_candidates,args.optimisation_iters,args.top_candidates,args.num_modes,args.action_std,reward_measure,args.device)
    else:
        raise ValueError("Planner argument not recognised")

    test_mpc(env, mpc_agent, logger,args.num_episodes,args.logdir, args.savedir)


if __name__ == '__main__':
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    print("Initialized")
    #parsing arguments
    #Experiment
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--savedir", type=str, default="save")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--num_seed_episodes", type=int, default=10)
    parser.add_argument("--num_collect_episodes", type=int, default=1)
    parser.add_argument("--num_test_episodes", type=int, default=1)
    parser.add_argument("--train_every", type=int, default=1)
    parser.add_argument("--test_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--num_warm_up_episodes", type=int, default=-1)
    #Environment
    parser.add_argument("--env_name", type=str, default="piche")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--action_repeat", type=int, default=3)
    #Model
    parser.add_argument("--hidden_size", type=int, default=350)
    parser.add_argument("--ensemble_size", type=int, default=1)
    #Training
    parser.add_argument("--warm_start", type=boolcheck, default=True)
    parser.add_argument("--model_buffer_size", type=int, default=1000000)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--model_batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--grad_clip_norm", type=int, default=1000)
    #Planning 
    parser.add_argument("--planner", type=str, default="cem")
    parser.add_argument("--plan_horizon", type=int, default=20)
    parser.add_argument("--optimisation_iters", type=int, default=5)
    parser.add_argument("--num_candidates", type=int, default=500)
    parser.add_argument("--top_candidates", type=int, default=50)
    parser.add_argument("--action_noise", type=float, default=0.3)
    parser.add_argument("--action_std", type=float, default=1)
    parser.add_argument("--cem_alpha", type=float, default=0.0)
    parser.add_argument("--discount_factor", type=float, default=1.0)
    #MPPI planner
    parser.add_argument("--lambda_", type=float, default=1)
    parser.add_argument("--noise_mu", type=float, default=0)
    parser.add_argument("--noise_sigma", type=float, default=1)
    #multimodal CEM
    parser.add_argument("--num_modes", type=int, default=3)
    args = parser.parse_args()
    args.device = DEVICE
    print("Args parsed")
    #create folders
    if args.savedir != "":
        subprocess.call(["mkdir","-p",str(args.savedir)])
    if args.logdir != "":
        subprocess.call(["mkdir","-p",str(args.logdir)])
    print("folders created")
    main(args)

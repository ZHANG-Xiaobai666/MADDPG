
import time
import numpy as np
import torch
from algorithms.agent import Agent
import os
import collections
import random
import copy





class EnvRunner():
    def __init__(self, args, env):
        #self.max_train_times = args.max_train_times
        self.episodes = args.episode_num
        self.episode_length = args.episode_length
        #self.reward_type = args.reward_type
        self.env = env


        self.num_agent = args.num_agent
        self.num_channel = args.num_channel
        self.gamma = args.gamma

        self.save_dir = args.save_dir

        self.batch_size = args.batch_size
        self.minimal_train_size = args.minimal_train_size
        self.learning_interval = args.learning_interval
        self.update_interval = args.update_interval
        self.buffer_size = args.buffer_size

        self.agents = []
        for _ in range(self.num_agent):
            agent = Agent(args)
            self.agents.append(agent)

        self.buffers = collections.deque(maxlen=self.buffer_size)



    def run(self):

        start = time.time()


        for episode in range(self.episodes):

            self.env.reset()
            obs_next = self.env.get_obs_his()
            obs = copy.deepcopy(obs_next)

            for step in range(self.episode_length):

                actions = self.collect(obs)             # collect actions and corresponding probs
                obs_next, actions_vector, rewards = self.env.step(actions, step)

                self.push(obs, actions_vector, rewards, obs_next)
                obs = copy.deepcopy(obs_next)
                if len(self.buffers) >= self.minimal_train_size and step % self.learning_interval == 0:
                    self.train()
                    if step % self.update_interval == 0:
                        self.eval_update()
                    if step % 1000 == 0:
                        print(f"Iteration: {step + 1} / {self.episode_length}")
                        print(f"Throughput {self.env.get_short_term_throughput(step)}")




            self.save_model()
            #.update_par(step, self.episode_length)





        end = time.time()
    def collect(self, obs):
        actions = []
        for agent in range(self.num_agent):
            action = self.agents[agent].select_action(obs[agent])
            actions.append(action)
        return actions

    def push(self, obs, actions, rewards, obs_next):
        self.buffers.append((obs, actions, rewards, obs_next))


    def train(self):
        buffers_sample = random.sample(self.buffers, self.batch_size)
        obs, actions, rewards, obs_next = zip(*buffers_sample)
        obs_next = torch.tensor(np.array(obs_next), dtype=torch.float32).reshape(self.batch_size, self.num_agent, 1, -1)  # (batch_size, num_agent,Seq = 1, obs_dim)
        obs = torch.tensor(np.array(obs), dtype=torch.float32).reshape(self.batch_size, self.num_agent, 1, -1)  # (batch_size, num_agent,Seq = 1, obs_dim)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).reshape(self.batch_size, self.num_agent, 1, -1)

        actions_next = []
        for agent in range(self.num_agent):
            action_next = self.agents[agent].select_action_from_eval(obs_next[:, agent, :, :])  # input (batch_size, 1, obs_dim)
            actions_next.append(action_next)                                              # output (batch_size, 1, 2)
        actions_next = torch.cat(actions_next, dim=-1)   # output (batch_size, 1, 2*num_agent)


        """critic network"""
        common_eval_critic_input = torch.cat((obs_next.reshape(obs_next.shape[0], 1, -1), actions_next), dim=-1)  # (batch_size, seq_len=1,xx)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).reshape(self.batch_size, 1, -1)
        common_critic_input = torch.cat((obs.reshape(obs.shape[0], 1, -1), actions), dim=-1)
        for agent in range(self.num_agent):
            self.agents[agent].train_critic(common_critic_input, rewards[:, agent, :, :], common_eval_critic_input)



        """actor network"""
        actions_from_trained = []
        q_values = []     #used for regularization
        for agent in range(self.num_agent):
            action_from_trained, q_value = self.agents[agent].select_action_from_trained(obs[:, agent, :, :])  # input (batch_size, 1, obs_dim)
            actions_from_trained.append(action_from_trained)
            q_values.append(q_value)


        """Only the action from current actor needs the gradient."""
        for agent in range(self.num_agent):
            new_actions_from_trained = [x if k == agent else x.detach() for k, x in enumerate(actions_from_trained)]
            new_actions_from_trained = torch.cat(new_actions_from_trained, dim=-1)
            critic_input = torch.cat((obs.reshape(obs.shape[0], 1, -1), new_actions_from_trained), dim=-1)
            self.agents[agent].train_actor(critic_input, q_values[agent])


    def eval_update(self):
        for agent in range(self.num_agent):
            self.agents[agent].soft_update()

    def save_model(self):
        for agent in range(self.num_agent):
            self.agents[agent].save(os.path.join(self.save_dir, "agent" + str(agent) + ".pt"))

    def load_model(self):
        for agent in range(self.num_agent):
            self.agents[agent].load(os.path.join(self.save_dir, "agent" + str(agent) + ".pt"))


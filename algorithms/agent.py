
import torch.nn.functional as F
import torch
import torch.nn as nn
from algorithms.actor import Actor
from algorithms.critic import Critic
import numpy as np
import copy

import math

class Agent(nn.Module):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.device = args.dvc
        self.num_channel = args.num_channel
        self.num_agent = args.num_agent


        self.actor_input_dim = int(args.obs_dim * args.obs_his_len)
        self.actor_net_width = args.actor_net_width
        self.actor_output_dim = args.action_dim

        self.actor = Actor(self.actor_input_dim, self.actor_net_width, self.actor_output_dim)
        self.actor_lr = args.actor_lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_lr)

        self.actor_eval = Actor(self.actor_input_dim, self.actor_net_width, self.actor_output_dim)
        self.actor_eval.load_state_dict(self.actor_eval.state_dict())

        """The centralized Q value is the function of Q(obs of all nodes, actions of all nodes) """
        self.critic_input_dim = self.actor_input_dim * self.num_agent + self.num_agent * args.action_dim
        self.critic_net_width = args.critic_net_width
        self.critic_output_dim = 1

        self.critic = Critic(self.critic_input_dim, self.critic_net_width, self.critic_output_dim)
        self.critic_lr = args.critic_lr
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_lr)

        self.critic_eval = Critic(self.critic_input_dim, self.critic_net_width, self.critic_output_dim)
        self.critic_eval.load_state_dict(self.critic_eval.state_dict())



        self.temperature = args.temperature
        self.epsilon = args.epsilon      # exploration probability
        self.gamma = args.gamma
        self.tau = args.tau
        #self.tpdv = dict(dtype=torch.float32, device=device)



    def select_action(self, obs, eps=1e-20, deterministic=False):
        #obs = check(obs).to(**self.tpdv)

        obs = torch.tensor(np.array(obs), dtype=torch.float32).view(1, 1, self.actor_input_dim)

        with torch.no_grad():
            q_value = self.actor(obs)

        if deterministic:
            """Note that in the provided code, it's a epsilon-greedy(0.01) selection. 
             Here, it is fully greedy.
             """
            action = F.gumbel_softmax(q_value, tau=self.temperature, hard=True, dim=-1)
            action = action.argmax(dim=-1)
            return action.item()
            # if np.random.rand() < self.epsilon:
            #     action = torch.randint(0, self.actor_output_dim, (1, 1, 1))
            #     return action.item()
            # else:
            #     return q_value.argmax(dim=-1).item()
        else:
            if np.random.rand() < self.epsilon:
                action = torch.randint(0, self.actor_output_dim, (1, 1, 1))
                return action.item()
            else:
                action = F.gumbel_softmax(q_value, tau=self.temperature, hard=True, dim=-1)
                action = action.argmax(dim=-1)
                return action.item()

    def select_action_from_eval(self, obs):  # (batch_size, seq_len=1, input_dim)
        with torch.no_grad():
            q_values = self.actor_eval(obs)
            actions = F.gumbel_softmax(q_values, tau=self.temperature, hard=True, dim=-1) # actions: (batch_size, seq_len=1, 1)
        return actions  # actions: (batch_size, seq_len=1, 2)

    """
    Q values are returned for regularization
    """
    def select_action_from_trained(self, obs):
        q_values = self.actor(obs)               # actions: (batch_size, seq_len=1, 2)
        actions = F.gumbel_softmax(q_values, tau=self.temperature, hard=True, dim=-1)
        #actions = q_values.argmax(dim=-1, keepdim=True)
        #x_seq = actions.squeeze(dim=-1)    # get the last dim of actions
        return actions, q_values         # (batch_size, seq_len=1, 2)


    def train_critic(self, common_critic_input, rewards, common_eval_critic_input): # (batch_size, seq_len=1, xx)
        with torch.no_grad():
            target_value = rewards + self.gamma * self.critic_eval(common_eval_critic_input)

        actual_value = self.critic(common_critic_input)

        loss = F.mse_loss(target_value, actual_value)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()


    def train_actor(self, critic_input, q_values):
        actor_loss = -self.critic(critic_input).mean()
        actor_loss += (q_values ** 2).mean() * 1e-3    # Regularization
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    #@torch.no_grad()
    def soft_update(self):
        for tp, sp in zip(self.critic_eval.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + sp.data * self.tau)

        for tp, sp in zip(self.actor_eval.parameters(), self.actor.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + sp.data * self.tau)


    def save(self, save_dir):
        torch.save(self.actor.state_dict(), save_dir)

    def load(self, load_dir):
        self.actor.load_state_dict(torch.load(load_dir, weights_only=True,  map_location=self.device))
        self.actor.eval()


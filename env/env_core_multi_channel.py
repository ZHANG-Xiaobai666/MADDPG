import random
import numpy as np

class EnvCore:

    def __init__(self, args, arr_pro):
        self.episode_length = args.episode_length       # Number of time slot per train/execute
        self.num_agent = args.num_agent                 # Number of nodes
        self.arr_pro = arr_pro                          # aggregate arrival probability
        self.num_channel = args.num_channel             # Number of channels

        self.alpha = args.alpha                         #reward balance coefficient

        #self.channel_feedback = [0 for _ in range(self.num_channel)]  # 1 if ACK 0 Otherwise

        self.queue_length = np.zeros(self.num_agent)   # backlogged packets
        #self.generate_time = [[] for _ in range(self.num_agent)] # For derivation of delay

        self.num_successful_packets = np.zeros(self.num_agent) # successful transmitted packets
        #self.mean_delay = [0 for _ in range(self.num_agent)]             # mean queueing delay of data packets
        self.throughput = 0                                             # network throughput sum(
                                                                        # num_successful_packets)/episode_length)

        self.obs_his_len = args.obs_his_len
        self.obs_obs_dim = args.obs_dim
        self.obs = []
        self.vl = []
        self.vl_ = []
        for _ in range(self.num_channel):
            _obs = [[np.zeros(self.obs_obs_dim) for _ in range(self.obs_his_len)] for _ in range(self.num_agent)]
            _vl = np.zeros(self.num_agent)
            _vl_ = np.zeros(self.num_agent)
            self.obs.append(_obs)
            self.vl.append(_vl)
            self.vl_.append(_vl_)


    def reset(self):
        #self.channel_state = [1 for _ in range(self.num_channel)]  # 0:idle, 1:successful 2:collision;

        self.queue_length = np.zeros(self.num_agent)    #  backlogged packets
        #self.generate_time = [[] for _ in range(self.num_agent)]  #  queue buffer
        self.num_successful_packets = np.zeros(self.num_agent)
        #self.mean_delay = [0 for _ in range(self.num_agent)]
        self.throughput = 0

        self.obs = []
        self.vl = []
        self.vl_ = []
        for _ in range(self.num_channel):
            _obs = [[np.zeros(self.obs_obs_dim) for _ in range(self.obs_his_len)] for _ in range(self.num_agent)]
            _vl = np.zeros(self.num_agent)
            _vl_ = np.zeros(self.num_agent)
            self.obs.append(_obs)
            self.vl.append(_vl)
            self.vl_.append(_vl_)


    def step(self, actions, time):  # actions NX1

        """ packet generation """
        random_numbers = np.array([random.random() for _ in range(self.num_agent)])
        index = np.where(random_numbers < self.arr_pro)[0]
        for idx in index:
            #self.generate_time[idx].append(time)
            self.queue_length[idx] += 1



        for ch in range(self.num_channel):
            nodes_feedback = np.zeros(self.num_agent)
            rewards_global = np.zeros(self.num_agent)
            rewards_ind = np.zeros(self.num_agent)

            # cal Dl(phi in the paper)
            if sum(self.vl[ch]) == 0:
                Dl = np.zeros(self.num_agent)
            else:
                Dl = self.vl[ch] / sum(self.vl[ch])

            # update vl  and vl_
            self.vl[ch] += 1
            self.vl_[ch] += 1

            actions_ch = np.array(actions[ch])

            """ get feedback from the channel """
            x = np.where(actions_ch == 1)[0]
            if len(x) == 1:                          # successful if only one node transmits
                nodes_feedback = [1 for _ in range(self.num_agent)]
                nodes_feedback[x[0]] = 0
                self.num_successful_packets[x[0]] += 1
                #self.mean_delay[x[0]] += (self.generate_time[x[0]][0] - time + 1)/self.num_successful_packets[x[0]]
                #del self.generate_time[x[0]][0]
                self.queue_length[x[0]] -= 1

                # update vl and vl_
                self.vl[ch][x[0]] = 0
                tmp = self.vl_[ch][x[0]]
                self.vl_[ch] = np.zeros(self.num_agent)
                self.vl_[ch][x[0]] = tmp

            elif len(x) > 1:                   # collision
                nodes_feedback = [2 for _ in range(self.num_agent)]
            else:                            # idle
                nodes_feedback = [3 for _ in range(self.num_agent)]

            """ get rewards """
            pri_idx = np.where(Dl == max(Dl))[0]
            for agent in range(self.num_agent):
                if agent in pri_idx:
                    rewards_ind[agent] = 1 if actions_ch[agent] == 1 else -1/(1-Dl[agent])
                    if nodes_feedback[agent] == 0:
                        rewards_global = np.ones(self.num_agent)
                else:
                    rewards_ind[agent] = -1 if actions_ch[agent] == 1 else 1
                    if nodes_feedback[agent] == 0:
                        rewards_global = np.ones(self.num_agent) * Dl[agent]
                if nodes_feedback[agent] == 2:
                    rewards_global[agent] = -1

            """
            The algorithm is different from the description in the paper: action is not a scalar but a vector [1, 0] -> 0; [0, 1] -> 1
            feedback is not a scalar but a vector [1, 0, 0, 0] -> 1;  [0, 1, 0, 0] -> 2...;
            update obs history
            """

            for agent in range(self.num_agent):
                action = np.array([1, 0]) if actions[ch][agent] == 0 else np.array([0, 1])
                node_feedback = np.zeros(4)
                node_feedback[nodes_feedback[agent]] = 1
                d_l1 = np.array([self.vl[ch][agent]/(self.vl[ch][agent]+self.vl_[ch][agent]+1e-20)])
                d_l1_ = 1 - d_l1
                new_obs = np.concatenate([action, node_feedback, d_l1, d_l1_])
                self.obs[ch][agent] = self.obs[ch][agent][1:] + [new_obs]


        if time == self.episode_length-1:
            self.throughput = sum(self.num_successful_packets)/self.episode_length

        return self.obs

    def get_throughput(self):
        return self.throughput

    def get_sum_success(self):
        return self.num_successful_packets

    def get_short_term_throughput(self, time):
        return sum(self.num_successful_packets)/(time + 1)

    def get_obs_his(self):
        return self.obs

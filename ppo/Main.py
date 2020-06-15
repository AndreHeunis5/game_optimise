
import numpy as np
from collections import deque
from typing import Dict, List
import torch
from torch.distributions import Categorical

from ppo.Trainer import Trainer
from ppo.Model import Model
from ppo.Worker import Worker

softmax = torch.nn.Softmax(dim=1)


class Main(object):
    """
    Runs the training loop
    """

    def __init__(self, device):
        self.device = device

        # #### Configurations

        # Game specific config
        num_players = 4
        player_state_size = 20
        self.state_size: int = num_players * player_state_size
        self.action_size: int = 20

        # gamma and lambda for advantage calculation
        self.gamma = 0.99
        self.lamda = 0.95

        # number of training steps to perform
        self.updates = 200  # 10000

        # number of epochs to train the model with sampled data
        self.epochs = 4
        # number of worker processes
        self.n_workers = 8
        # number of steps to run on each process for a single update
        self.worker_steps = 1024
        # number of mini batches
        self.n_mini_batch = 8
        # total number of samples for a single update
        self.batch_size = self.n_workers * self.worker_steps
        # size of a mini batch
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        assert (self.batch_size % self.n_mini_batch == 0)

        # #### Initialize

        # create workers
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]

        # initialize tensors for observations and action masks
        self.obs = np.zeros((self.n_workers, self.state_size), dtype=np.uint8)
        self.action_mask = np.zeros((self.n_workers, self.action_size), dtype=np.bool)
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i], self.action_mask[i] = worker.child.recv()

        # model for sampling
        self.model = Model(state_size=self.state_size, action_size=self.action_size)
        self.model.to(device)
        # trainer
        self.trainer = Trainer(self.model)

    def run_training_loop(self):
        """
        ### Run training loop
        """
        # last 100 episode information
        episode_info = deque(maxlen=100)
        rewards = np.zeros(self.updates)
        gamelength = np.zeros(self.updates)

        for update in range(self.updates):
            progress = update / self.updates

            # decreasing `learning_rate` and `clip_range` epsilon
            learning_rate = 2.5e-4 * (1 - progress)
            clip_range = 0.1 * (1 - progress)

            # sample with current policy
            samples, sample_episode_info = self._sample_from_workers()

            # train the model
            self.train(samples, learning_rate, clip_range)

            # collect episode info
            episode_info.extend(sample_episode_info)

            # mean of last 100 episodes
            reward_mean, length_mean = Main._get_mean_episode_info(episode_info)
            rewards[update] = reward_mean
            gamelength[update] = length_mean

            # write summary info to the writer, and log to the screen
            print(f"{update:4}: reward={reward_mean:.2f} length={length_mean:.3f}")

        return rewards, gamelength

    def _sample_from_workers(self) -> (Dict[str, np.ndarray], List):
        """
        Sample data with current policy

        :return:    Dict. samples for training
                    List. info on completed games
        """
        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        # TODO shouldnt have game specific stuff here
        take_second_turns = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        obs = np.zeros((self.n_workers, self.worker_steps, self.state_size), dtype=np.uint8)
        neg_log_pis = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        values = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        episode_infos = []

        # sample `worker_steps` from each worker
        for t in range(self.worker_steps):
            # `self.obs` keeps track of the last observation from each worker,
            #  which is the input for the model to sample the next action
            obs[:, t] = self.obs
            # sample actions from the old policy for each worker;
            #  this returns arrays of size `n_workers`
            pi_logits, v = self.model(torch.from_numpy(self.obs).float().to(self.device))
            pi_probs = softmax(pi_logits)

            #  masking out invalid actions and renormalising
            pi_probs[torch.from_numpy(~self.action_mask)] = 0
            pi_probs /= pi_probs.sum(axis=1, keepdims=True)
            pi = Categorical(probs=pi_probs)

            #  value
            values[:, t] = v.cpu().data.numpy()

            #  action
            a = pi.sample()
            actions[:, t] = a.cpu().data.numpy()
            # TODO should i be using the masked policy or unmasked?. currently using masked
            neg_log_pis[:, t] = -pi.log_prob(a).cpu().data.numpy()

            # run sampled actions on each worker
            for w, worker in enumerate(self.workers):
                # TODO take second turn needs to be populated correctly
                worker.child.send(("step", {'action_to_take': actions[w, t], 'take_second_turn': False}))

            for w, worker in enumerate(self.workers):
                # get results after executing the actions
                self.obs[w], rewards[w, t], dones[w, t], take_second_turns[w, t], self.action_mask[w], info = worker.child.recv()

                # collect episode info, which is available if an episode finished;
                #  this includes total reward and length of the episode -
                #  look at `Game` to see how it works.
                # We also add a game frame to it for monitoring.
                if info is not None:
                    info['obs'] = obs[w, t]
                    episode_infos.append(info)

        # calculate advantages
        advantages = self._calc_advantages(dones, rewards, values)
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values,
            'neg_log_pis': neg_log_pis,
            'advantages': advantages
        }

        # samples are currently in [workers, time] table, we should flatten it
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = torch.from_numpy(v).to(self.device)
            else:
                samples_flat[k] = torch.tensor(v, device=self.device)

        return samples_flat, episode_infos

    def _calc_advantages(self, dones: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        # advantages table
        advantages = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0

        # Value(s_t+1)
        _, last_value = self.model(torch.from_numpy(self.obs).float().to(self.device))
        last_value = last_value.cpu().data.numpy()

        for t in reversed(range(self.worker_steps)):
            # mask if episode completed after step t
            mask = 1.0 - dones[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            # delta_t
            delta = rewards[:, t] + self.gamma * last_value - values[:, t]

            # hat A_t
            last_advantage = delta + self.gamma * self.lamda * last_advantage

            # note that we are collecting in reverse order.
            advantages[:, t] = last_advantage
            last_value = values[:, t]

        return advantages

    def train(self, samples: Dict[str, np.ndarray], learning_rate: float, clip_range: float):
        """
        ### Train the model based on samples
        """
        # collect training information like losses for monitoring
        train_info = []

        # higher number of epochs -> faster learning but less stability (could be solved by reducing clipping range?)
        for _ in range(self.epochs):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            # for each mini batch
            for start in range(0, self.batch_size, self.mini_batch_size):
                # get mini batch
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                # train
                res = self.trainer.train(learning_rate=learning_rate,
                                         clip_range=clip_range,
                                         samples=mini_batch)

                # append to training information
                train_info.append(res)

        # return average of training information
        return np.mean(train_info, axis=0)

    @staticmethod
    def _get_mean_episode_info(episode_info):
        """
        #### Get average episode reward and episode length
        """
        if len(episode_info) > 0:
            return (np.mean([info["reward"] for info in episode_info]),
                    np.mean([info["game_length"] for info in episode_info]))
        else:
            return np.nan, np.nan

    def destroy(self):
        """
        ### Destroy
        Stop the workers
        """
        for worker in self.workers:
            worker.child.send(("close", None))
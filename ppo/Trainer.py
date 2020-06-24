
import numpy as np
import torch
from typing import Dict
from torch import optim

from ppo.Model import Model
from torch.distributions import Categorical


class Trainer:

    def __init__(self, model: Model):
        """
        ### Initialization
        """
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

    def train(self, samples: Dict[str, np.ndarray], learning_rate: float, clip_range: float):
        """

        :param samples:         observations, actions, values, advantages, neg_log_pis
        :param learning_rate:
        :param clip_range:
        :return:
        """
        sampled_obs = samples['obs']

        # actions / values / advantages sampled from the old policy
        sampled_action = samples['actions']
        sampled_return = samples['values'] + samples['advantages']

        # z-score normalise
        sampled_normalized_advantage = Trainer._normalize(samples['advantages'])

        sampled_neg_log_pi = samples['neg_log_pis']
        sampled_value = samples['values']

        # #### Policy

        #TODO using masking in game playing but not here?
        pi_logits, value = self.model(sampled_obs.float())
        pi = Categorical(logits=pi_logits)
        neg_log_pi = -pi.log_prob(sampled_action)

        # ratio of new theta / old theta
        ratio = torch.exp(sampled_neg_log_pi - neg_log_pi)

        # Using the normalized advantage introduces a bias to the policy gradient estimator, but it reduces variance a
        # lot.
        clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_normalized_advantage, clipped_ratio * sampled_normalized_advantage)
        policy_reward = policy_reward.mean()

        # #### Entropy Bonus

        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # #### Value

        clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip_range,
                                                                      max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = 0.5 * vf_loss.mean()

        # Full loss function
        loss: torch.Tensor = -(policy_reward - 0.5 * vf_loss + 0.01 * entropy_bonus)

        # compute gradients
        for pg in self.optimizer.param_groups:
            pg['lr'] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        # for monitoring
        approx_kl_divergence = .5 * ((neg_log_pi - sampled_neg_log_pi) ** 2).mean()
        clip_fraction = (abs((ratio - 1.0)) > clip_range).type(torch.FloatTensor).mean()

        return [policy_reward,
                vf_loss,
                entropy_bonus,
                approx_kl_divergence,
                clip_fraction]

    @staticmethod
    def _normalize(adv: np.ndarray):
        """#### Normalize advantage function"""
        return (adv - adv.mean()) / (adv.std() + 1e-8)

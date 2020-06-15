
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

        # model for training, $\pi_\theta$ and $V_\theta$.
        # This model shares parameters with the sampling model so,
        #  updating variables affect both.
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

    def train(self, samples: Dict[str, np.ndarray], learning_rate: float, clip_range: float):
        """

        :param samples:         observations, actions, values, advantages, neg_log_pis
        :param learning_rate:
        :param clip_range:
        :return:
        """
        # sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$;
        #  we are treating observations as state
        sampled_obs = samples['obs']

        # $a_t$ actions sampled from $\pi_{\theta_{OLD}}$
        sampled_action = samples['actions']
        # $R_t$ returns sampled from $\pi_{\theta_{OLD}}$
        sampled_return = samples['values'] + samples['advantages']

        # $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$,
        # where $\hat{A_t}$ is advantages sampled from $\pi_{\theta_{OLD}}$.
        # Refer to sampling function in [Main class](#main) below
        #  for the calculation of $\hat{A}_t$.
        sampled_normalized_advantage = Trainer._normalize(samples['advantages'])

        # $-\log \pi_{\theta_{OLD}} (a_t|s_t)$ log probabilities
        sampled_neg_log_pi = samples['neg_log_pis']
        # $\hat{V_t}$ value estimates
        sampled_value = samples['values']

        # `pi_logits` and $V^{\pi_\theta}(s_t)$
        pi_logits, value = self.model(sampled_obs.float())

        # #### Policy

        # $-\log \pi_\theta (a_t|s_t)$
        #TODO using masking in game playing but not here?
        pi = Categorical(logits=pi_logits)
        neg_log_pi = -pi.log_prob(sampled_action)

        # ratio $r_t(\theta) = \frac{\pi_\theta (a_t|s_t)}{\pi_{\theta_{OLD}} (a_t|s_t)}$;
        # *this is different from rewards* $r_t$.
        ratio: torch.Tensor = torch.exp(sampled_neg_log_pi - neg_log_pi)

        # Using the normalized advantage
        #  $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$
        #  introduces a bias to the policy gradient estimator,
        #  but it reduces variance a lot.
        clipped_ratio = ratio.clamp(min=1.0 - clip_range,
                                    max=1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_normalized_advantage,
                                  clipped_ratio * sampled_normalized_advantage)
        policy_reward = policy_reward.mean()

        # #### Entropy Bonus

        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # #### Value

        # Clipping makes sure the value function $V_\theta$ doesn't deviate
        #  significantly from $V_{\theta_{OLD}}$.
        clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip_range,
                                                                      max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = 0.5 * vf_loss.mean()

        # we want to maximize $\mathcal{L}^{CLIP+VF+EB}(\theta)$
        # so we take the negative of it as the loss
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
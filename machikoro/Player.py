
from machikoro.enums.IncomeType import IncomeType
from machikoro.Establishment import WheatField, Bakery
from common.PlayerType import PlayerType
from ppo.Model import Model

import numpy as np
import torch
from torch.distributions import Categorical

softmax = torch.nn.Softmax(dim=1)

if torch.cuda.is_available():
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")


class Player:

	def __init__(self, pos: int, player_type: PlayerType, action_dim: int, model_path: str = None):
		"""

		:param pos:
		:param player_type:
		:param model_path: 		If player_type is AGENT_TRAINED, path to the trained model parameters, otherwise None
		"""
		self.pos = pos
		self.coins = 3
		self.player_type = player_type
		self.action_dim = action_dim

		self.cards = {
			IncomeType.BLUE: [WheatField()],
			IncomeType.RED: [],
			IncomeType.GREEN: [Bakery()],
			IncomeType.PURPLE: []}

		self.landmarks = {
			'train_station': False,
			'shopping_mall': False,
			'amusement_park': False,
			'radio_tower': False}

		self.model = None
		if player_type == PlayerType.AGENT_TRAINED and model_path is not None:
			#TODO fetch sizes from single point of definition
			self.model = Model(state_size=80, action_size=self.action_dim)
			self.model.load_state_dict(torch.load(model_path + 'model_240.pt'))
			self.model.eval()
			self.model.to(device)
		elif player_type == PlayerType.AGENT_TRAINED and model_path is None:
			raise ValueError("Need a pretrained model for AGENT_TRAINED")

	def get_action(self, action_mask: np.array, to_buy=None, state: np.array=None) -> int:
		"""
		Generate a valid action for the player.

		If the player type is AGENT_INTRAINING, the action will be generated by the training code. This function will
		return an error if called on an AGENT_INTRAINING player.

		:param state: 				Not required if player_type is RANDOMAGENT
		:param action_mask:
		:return:
		"""
		if self.player_type == PlayerType.RANDOMAGENT:
			return self._get_random_action(action_mask=action_mask)
		elif self.player_type == PlayerType.AGENT_TRAINED:
			return self._get_agent_action(state=state, action_mask=action_mask)
		elif self.player_type == PlayerType.HUMAN:
			return self._get_human_action(state=state, action_mask=action_mask, to_buy=to_buy)
		else:
			raise ValueError("get_action does not support player type: {}".format(self.player_type))

	def _get_random_action(self, action_mask: np.array) -> int:
		random_action_probs = np.random.rand(self.action_dim)
		random_action_probs[~action_mask] = 0
		random_action_probs = random_action_probs / np.sum(random_action_probs)  # renormalise to sum to 1
		random_action = np.random.choice(range(self.action_dim), p=random_action_probs)

		return random_action

	def _get_agent_action(self, state: np.array, action_mask: np.array) -> int:
		pi_logits, v = self.model(torch.from_numpy(state).float().to(device))
		pi_logits = torch.reshape(pi_logits, (1, -1))
		pi_probs = softmax(pi_logits)

		#  masking out invalid actions and renormalising
		pi_probs[torch.from_numpy(~action_mask.reshape((1, -1)))] = 0
		pi_probs /= pi_probs.sum(axis=1, keepdims=True)
		pi = Categorical(probs=pi_probs)
		action = pi.sample()
		print('ai took action {}'.format(action))
		return action

	def _get_human_action(self, state: np.array, action_mask: np.array, to_buy) -> int:

		unnorm = np.copy(state)
		for i in range(0, 80, 20):
			unnorm[i] = unnorm[i] * 50.0
			unnorm[i+1:i+13] = unnorm[i+1:i+13] * 6.0
			unnorm[i+13:i+16] = unnorm[i+13:i+16] * 4.0

		print('----HUMAN TURN----')
		print('Available to buy: ', to_buy)
		print('Player state: ', unnorm[:20])
		print('Agent state: ', unnorm[20:40])
		print('Agent state: ', unnorm[40:60])
		print('Agent state: ', unnorm[60:])

		valid = False
		while not valid:
			test = int(input("Enter your action index: "))
			valid = action_mask[test]

			if not valid:
				print('Invalid action selected')
		print('Selected: {}'.format(test))

		return test

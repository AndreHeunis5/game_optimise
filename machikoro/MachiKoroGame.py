"""
# Action Space

* 0         do nothing
* 1 - 5     buy blue
    * wheatfield, ranch, forest, mine, apple orchard
* 6 - 10     buy green
    * bakery, convenience store, cheese factory, furniture factory, farmers market
* 11 - 12   buy red
    * cafe, family restaurant
* 13 - 15   buy purple
    * stadium, tv station, business center
* 16 - 19   buy landmark
    * train station, shopping mall, amusement park

20 total

## actions todo
* choosing a player for tv station activation
* choosing something to trade in business center
* whether to throw 1 or 2 dice (train station)
* whether to reroll doubles (amusement park)
* reroll (radio tower)

## State Space
For each player
* 0         their coin value
* 1 - 15    number of each establishment, normalised between 0 and the max
* 16 - 19   binary whether they activated a landmark

20 total

# Results

After 10k games the win proportions for each player taking random actions is array([0.2591, 0.2599, 0.2405, 0.2405])
"""
from copy import deepcopy

from machikoro.Establishment import *
from machikoro.Player import Player
from common.PlayerType import PlayerType

verbose = False


class MachiKoroGame:

	def __init__(self, player_types: list, trained_model_path: str = None):
		self.player_types = player_types
		self.trained_model_path = trained_model_path

		self.ACTION_DIM: int = int(20)
		self.num_players = 4
		self.PLAYER_STATE_DIM: int = 1 + 15 + 4
		self.landmark_cost = np.array([4, 10, 16, 22], dtype=int)
		self.all_establishments = [WheatField(), Ranch(), Forest(), Mine(), AppleOrchard(),
															 Bakery(), ConvenienceStore(), CheeseFactory(), FurnitureFactory(), FarmersMarket(),
															 Cafe(), FamilyRestaurant(),
															 Stadium(), TvStation(), BusinessCentre()]

		self.turn_count = None
		self.players = None
		self.establishments_available_to_buy = None
		self.reset_game(model_path=self.trained_model_path)

	def reset_game(self, model_path: str):
		self.turn_count = 0

		self.players = []
		for i, pt in enumerate(self.player_types):
			if pt == PlayerType.AGENT_TRAINED:
				self.players.append(Player(
					pos=i,
					player_type=pt,
					action_dim=self.ACTION_DIM,
					model_path=model_path))
			else:
				self.players.append(Player(pos=i, player_type=pt, action_dim=self.ACTION_DIM))

		self.establishments_available_to_buy = np.array([
			6, 6, 6, 6, 6,  # BLUE
			6, 6, 6, 6, 6,  # GREEN
			6, 6,  					# RED
			4, 4, 4  				# PURPLE
		])

		# the first dice roll should happen in init so that the step structure can be 1. pn decision -> 2. pn+1 rolls
		self._resolve_dice(active_player=self.players[0])

		obs = self._build_state_vector(ap=self.players[0].pos)
		action_mask = self._build_action_mask(active_player=self.players[0])

		return obs, action_mask

	def step(self, step_args: dict) -> (np.array, int, bool, bool):
		"""
		A step is defined as
			- The RL agent takes a decision (the given action)
			- all the random players roll dice and take their turns with random actions
			- The RL agent rolls the dice

		:param:		step_args:	 dict containing the keys
															action_to_take: int
															take_second_turn: bool
		:return:				obs
										reward
										done
										take_second_turn
										action_mask				np.array of bool indicating valid actions for the next step. None if done == True
										game_length				int. number of turns taken in current game
		"""
		action_to_take = step_args['action_to_take']
		take_second_turn = step_args['take_second_turn']
		self.turn_count += 1

		# First take the RL agent action
		done = self._resolve_active_player_actions(active_player=self.players[0], action_to_take=action_to_take)

		if done:
			if verbose: print('player 0 won')
			return self._build_state_vector(ap=0), 1 - self.turn_count / 50.0, done, False, None, self.turn_count

		# Skip other players if agent needs a second turn
		if not take_second_turn:
			# Resolve random actions
			for ap in self.players[1:]:
				self._resolve_dice(active_player=ap)
				action_mask = self._build_action_mask(active_player=ap)

				if ap.player_type == PlayerType.RANDOMAGENT:
					player_action = ap.get_action(action_mask=action_mask)
				elif ap.player_type == PlayerType.AGENT_TRAINED:
					state = self._build_state_vector(ap=ap.pos)
					player_action = ap.get_action(action_mask=action_mask, state=state)
				done = self._resolve_active_player_actions(active_player=ap, action_to_take=player_action)

				if done:
					if verbose: print('player {} won'.format(ap.pos))
					return self._build_state_vector(ap=ap.pos), -1 - self.turn_count / 50.0, done, False, None, self.turn_count

		# Last step is for the agent to roll the dice for its next turn
		is_double_roll = self._resolve_dice(active_player=self.players[0])
		take_second_turn = is_double_roll and not take_second_turn and self.players[0].landmarks['amusement_park']

		obs = self._build_state_vector(ap=0)
		action_mask = self._build_action_mask(active_player=self.players[0])

		return obs, 0, done, take_second_turn, action_mask, None

	def _resolve_active_player_actions(self, active_player: Player, action_to_take: int):
		"""

		:param active_player:
		:param active_player:
		:return:
		"""
		if verbose: print('Player {} has {} coin and takes action {}'.format(
			active_player.pos,
			active_player.coins,
			action_to_take))

		if action_to_take == 0:
			return False
		elif 0 < action_to_take < 16:
			bought_estab = deepcopy(self.all_establishments[action_to_take - 1])
			active_player.cards[bought_estab.type].append(bought_estab)
			active_player.coins -= bought_estab.cost
			self.establishments_available_to_buy[action_to_take - 1] -= 1
		else:  # TODO clean up this step
			if action_to_take == 16:
				active_player.landmarks['train_station'] = True
				active_player.coins -= self.landmark_cost[0]
			elif action_to_take == 17:
				active_player.landmarks['shopping_mall'] = True
				active_player.coins -= self.landmark_cost[1]
			elif action_to_take == 18:
				active_player.landmarks['amusement_park'] = True
				active_player.coins -= self.landmark_cost[2]
			elif action_to_take == 19:
				active_player.landmarks['radio_tower'] = True
				active_player.coins -= self.landmark_cost[3]

		return self._player_has_won(player_to_check=active_player)

	def _resolve_dice(self, active_player):
		dice_val, is_double_roll = self._roll_dice(active_player)
		self._resolve_dice_roll(active_player, dice_val)

		return is_double_roll

	# TODO randomly choosing 1 or 2 dice (Train station)
	# TODO choose to reroll (Radio Tower)
	def _roll_dice(self, active_player):
		"""

		:param active_player:
		:return: 	the total rolled value
							bool indicating if a double was rolled
		"""
		ndice = 1
		if active_player.landmarks['train_station']:
			ndice = np.random.choice(2)

		res = np.random.choice(6) + 1
		is_double_roll = False
		if ndice == 2:
			second_roll = np.random.choice(6) + 1
			is_double_roll = res == second_roll
			res += second_roll

		if verbose: print('Player {} rolled {}'.format(active_player.pos, res))

		return res, is_double_roll

	def _resolve_dice_roll(self, active_player, dice_val):
		"""
		From the game rules:

		It is possible that multiple types of Establishments are activated by the same die roll, in this case the
		Establishments are activated in the following order:
			Restaurants (Red)
			Secondary Industry (Green) and Primary Industry (Blue)
			Major Establishments (Purple)

		:param active_player:	The player whose turn it is
		:param dice_val: 			The value of the dice roll
		:return:
		"""
		# Red activations for all players other than the active player
		# Need to loop counter clockwise away from the active player to get the correct order of payments
		players_to_pay = self.players[:active_player.pos][::-1]
		players_to_pay.extend(self.players[active_player.pos + 1:][::-1])
		for p in players_to_pay:
			if p.pos == active_player.pos: continue
			for c in p.cards[IncomeType.RED]:
				if dice_val not in c.activation: continue
				c.effect(p, active_player, self.players)

		# Blue activations for everyone
		for p in self.players:
			for c in p.cards[IncomeType.BLUE]:
				if dice_val not in c.activation: continue
				c.effect(p, active_player, self.players)

		for c in active_player.cards[IncomeType.GREEN]:
			if dice_val not in c.activation: continue
			c.effect(active_player, active_player, self.players)

		# Purple activations
		for p in self.players:
			for c in p.cards[IncomeType.PURPLE]:
				if dice_val not in c.activation: continue
				c.effect(p, active_player, self.players)

	def _player_has_won(self, player_to_check) -> bool:

		if player_to_check.landmarks['train_station'] \
						and player_to_check.landmarks['shopping_mall'] \
						and player_to_check.landmarks['amusement_park'] \
						and player_to_check.landmarks['radio_tower']:
			return True
		else:
			return False

	def _build_state_vector(self, ap: int) -> np.array:
		"""
		State is built in player turn order starting with the active player

		:param ap:
		:return:
		"""
		state = np.zeros(self.num_players * self.PLAYER_STATE_DIM, dtype=np.float32)
		ind = 0

		player_turn_order = self.players[ap:] + self.players[:ap]
		for p in player_turn_order:

			state[ind] = p.coins

			state[ind + 1] = sum(map(lambda x: x.name == 'WheatField', p.cards[IncomeType.BLUE]))
			state[ind + 2] = sum(map(lambda x: x.name == 'Ranch', p.cards[IncomeType.BLUE]))
			state[ind + 3] = sum(map(lambda x: x.name == 'Forest', p.cards[IncomeType.BLUE]))
			state[ind + 4] = sum(map(lambda x: x.name == 'Mine', p.cards[IncomeType.BLUE]))
			state[ind + 5] = sum(map(lambda x: x.name == 'AppleOrchard', p.cards[IncomeType.BLUE]))

			state[ind + 6] = sum(map(lambda x: x.name == 'Bakery', p.cards[IncomeType.GREEN]))
			state[ind + 7] = sum(map(lambda x: x.name == 'ConvenienceStore', p.cards[IncomeType.GREEN]))
			state[ind + 8] = sum(map(lambda x: x.name == 'CheeseFactory', p.cards[IncomeType.GREEN]))
			state[ind + 9] = sum(map(lambda x: x.name == 'FurnitureFactory', p.cards[IncomeType.GREEN]))
			state[ind + 10] = sum(map(lambda x: x.name == 'FarmersMarket', p.cards[IncomeType.GREEN]))

			state[ind + 11] = sum(map(lambda x: x.name == 'Cafe', p.cards[IncomeType.RED]))
			state[ind + 12] = sum(map(lambda x: x.name == 'FamilyRestaurant', p.cards[IncomeType.RED]))

			state[ind + 13] = sum(map(lambda x: x.name == 'Stadium', p.cards[IncomeType.PURPLE]))
			state[ind + 14] = sum(map(lambda x: x.name == 'TvStation', p.cards[IncomeType.PURPLE]))
			state[ind + 15] = sum(map(lambda x: x.name == 'BusinessCentre', p.cards[IncomeType.PURPLE]))

			state[ind + 16] = int(p.landmarks['train_station'])
			state[ind + 17] = int(p.landmarks['shopping_mall'])
			state[ind + 18] = int(p.landmarks['amusement_park'])
			state[ind + 19] = int(p.landmarks['radio_tower'])

			ind += self.PLAYER_STATE_DIM

		state = self.process_obs(state)

		return state

	def _build_action_mask(self, active_player) -> np.array:
		# mask invalid actions
		action_mask = np.array([True] * self.ACTION_DIM)

		# ---- Establishments ----
		is_available = self.establishments_available_to_buy > 0
		action_mask[1:16] = np.logical_and(action_mask[1:16], is_available)

		can_afford_establishment = np.array([active_player.coins >= e.cost for e in self.all_establishments])
		action_mask[1:16] = np.logical_and(action_mask[1:16], can_afford_establishment)

		# ---- Landmarks ----
		can_afford_landmark = active_player.coins >= self.landmark_cost
		action_mask[16:] = np.logical_and(action_mask[16:], can_afford_landmark)

		landmark_available_to_flip = np.array([active_player.landmarks['train_station'],
																					 active_player.landmarks['shopping_mall'],
																					 active_player.landmarks['amusement_park'],
																					 active_player.landmarks['radio_tower']])
		action_mask[16:] = np.logical_and(action_mask[16:], ~landmark_available_to_flip)

		return action_mask

	def process_obs(self, obs):
		"""
		Normalise the state for input to PPO models

		:param obs:
		:return:
		"""
		for i in range(0, 80, 20):
			obs[i] = obs[i] / 50.0
			obs[i+1:i+13] = obs[i+1:i+13] / 6.0
			obs[i+13:i+16] = obs[i+13:i+16] / 4.0
		return obs

	def playtest_step(self, to_buy):
		done = False

		for ap in self.players:
			is_double_roll = self._resolve_dice(active_player=ap)

			state = None
			if ap.player_type != PlayerType.RANDOMAGENT:
				state = self._build_state_vector(ap=ap.pos)
			action_mask = self._build_action_mask(active_player=ap)

			action = ap.get_action(action_mask=action_mask, state=state, to_buy=to_buy)
			done = self._resolve_active_player_actions(active_player=ap, action_to_take=action)

			if done:
				print('Player {} won!'.format(ap.pos))
				return done

			if is_double_roll and ap.landmarks['amusement_park']:

				if ap.player_type == PlayerType.HUMAN:
					print('Second turn from double roll')

				if ap.player_type != PlayerType.RANDOMAGENT:
					state = self._build_state_vector(ap=ap.pos)
				action_mask = self._build_action_mask(active_player=ap)

				action = ap.get_action(action_mask=action_mask, state=state, to_buy=to_buy)
				done = self._resolve_active_player_actions(active_player=ap, action_to_take=action)

				if done:
					print('Player {} won!'.format(ap.pos))
					return done

		return done

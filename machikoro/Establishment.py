
from abc import ABC, abstractmethod
import numpy as np

from machikoro.IncomeType import IncomeType
from machikoro.Icon import Icon
from machikoro.utils import get_coins_for_icon, get_coins_from_active_player


class Establishment(ABC):

	def __init__(self, name: str, type: IncomeType, icon: Icon, cost: int, activation: list):
		self.name = name
		self.type = type
		self.icon = icon
		self.cost = cost
		self.activation = activation

	@abstractmethod
	def effect(self, player, roller, players):
		pass


# ------------------------------- BLUE -------------------------------

class WheatField(Establishment):
	def __init__(self):
		super().__init__(name='WheatField', type=IncomeType.BLUE, icon=Icon.WHEAT, cost=1, activation=[1])

	def effect(self, player, roller, players):
		player.coins += 1


class Ranch(Establishment):
	def __init__(self):
		super().__init__(name='Ranch', type=IncomeType.BLUE, icon=Icon.LIVESTOCK, cost=1, activation=[2])

	def effect(self, player, roller, players):
		player.coins += 1


class Forest(Establishment):
	def __init__(self):
		super().__init__(name='Forest', type=IncomeType.BLUE, icon=Icon.COG, cost=3, activation=[5])

	def effect(self, player, roller, players):
		player.coins += 1


class Mine(Establishment):
	def __init__(self):
		super().__init__(name='Mine', type=IncomeType.BLUE, icon=Icon.COG, cost=6, activation=[9])

	def effect(self, player, roller, players):
		player.coins += 5


class AppleOrchard(Establishment):
	def __init__(self):
		super().__init__(name='AppleOrchard', type=IncomeType.BLUE, icon=Icon.WHEAT, cost=3, activation=[10])

	def effect(self, player, roller, players):
		player.coins += 3

# ------------------------------- GREEN -------------------------------

class Bakery(Establishment):
	def __init__(self):
		super().__init__(name='Bakery', type=IncomeType.GREEN, icon=Icon.BREAD, cost=1, activation=[2, 3])

	def effect(self, card_owner, active_player, players):
		active_player.coins += 1
		if active_player.landmarks['shopping_mall']:
			active_player.coins += 1


class ConvenienceStore(Establishment):
	def __init__(self):
		super().__init__(name='ConvenienceStore', type=IncomeType.GREEN, icon=Icon.BREAD, cost=2, activation=[4])

	def effect(self, card_owner, active_player, players):
		active_player.coins += 3
		if active_player.landmarks['shopping_mall']:
			active_player.coins += 1


class CheeseFactory(Establishment):
	def __init__(self):
		super().__init__(name='CheeseFactory', type=IncomeType.GREEN, icon=Icon.FACTORY, cost=5, activation=[7])

	def effect(self, card_owner, active_player, players):
		get_coins_for_icon(player=active_player, icon=Icon.LIVESTOCK, amount_per_est=3)


class FurnitureFactory(Establishment):
	def __init__(self):
		super().__init__(name='FurnitureFactory', type=IncomeType.GREEN, icon=Icon.FACTORY, cost=3, activation=[8])

	def effect(self, card_owner, active_player, players):
		get_coins_for_icon(player=active_player, icon=Icon.COG, amount_per_est=3)


class FarmersMarket(Establishment):
	def __init__(self):
		super().__init__(name='FarmersMarket', type=IncomeType.GREEN, icon=Icon.APPLE, cost=2, activation=[11, 12])

	def effect(self, card_owner, active_player, players):
		get_coins_for_icon(player=active_player, icon=Icon.WHEAT, amount_per_est=2)

# ------------------------------- RED -------------------------------

class Cafe(Establishment):

	def __init__(self):
		super().__init__(name='Cafe', type=IncomeType.RED, icon=Icon.CUP, cost=2, activation=[3])

	def effect(self, card_owner, active_player, players):
		coins_to_gain = 1
		if card_owner.landmarks['shopping_mall']:
			coins_to_gain = 2
		get_coins_from_active_player(card_owner, active_player, coins_to_gain)


class FamilyRestaurant(Establishment):
	def __init__(self):
		super().__init__(name='FamilyRestaurant', type=IncomeType.RED, icon=Icon.CUP, cost=3, activation=[9, 10])

	def effect(self, card_owner, active_player, players):
		coins_to_gain = 2
		if card_owner.landmarks['shopping_mall']:
			coins_to_gain = 3
		get_coins_from_active_player(card_owner, active_player, coins_to_gain)

# ------------------------------- PURPLE -------------------------------


class Stadium(Establishment):
	def __init__(self):
		super().__init__(name='Stadium', type=IncomeType.PURPLE, icon=Icon.TOWER, cost=6, activation=[6])

	def effect(self, player, roller, players):
		total = 0

		for p in players:
			amount_to_take = min(p.coins, 2)
			p.coins -= amount_to_take
			total += amount_to_take

		player.coins += total


# TODO taking from a random player
class TvStation(Establishment):
	def __init__(self):
		super().__init__(name='TvStation', type=IncomeType.PURPLE, icon=Icon.TOWER, cost=7, activation=[6])

	def effect(self, card_owner, active_player, players):
		player_to_take_from = np.random.choice(4)
		num_coins_taken = min(3, players[player_to_take_from].coins)
		players[player_to_take_from].coins -= num_coins_taken
		active_player.coins += num_coins_taken


class BusinessCentre(Establishment):
	def __init__(self):
		super().__init__(name='BusinessCentre', type=IncomeType.PURPLE, icon=Icon.TOWER, cost=8, activation=[6])

	# TODO implement swapping establishments
	def effect(self, card_owner, active_player, players):
		pass
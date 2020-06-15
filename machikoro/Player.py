
from machikoro.IncomeType import IncomeType
from machikoro.Establishment import WheatField, Bakery


class Player:

	def __init__(self, pos):
		self.pos = pos
		self.coins = 3

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
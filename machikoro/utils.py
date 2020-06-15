
from machikoro.Icon import Icon


def get_coins_from_active_player(card_owner, active_player, coins):
	amount_to_pay = min(coins, active_player.coins)
	active_player.coins -= amount_to_pay
	card_owner.coins += amount_to_pay


def get_coins_for_icon(player, icon: Icon, amount_per_est: int):
	count = 0
	for card_type in player.cards.items():
		for c in card_type:
			if c.icon == icon:
				count += 1
	player.coins += count * amount_per_est
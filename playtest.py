
from machikoro.MachiKoroGame import MachiKoroGame
from common.PlayerType import PlayerType

if __name__ == "__main__":

	player_types = [PlayerType.AGENT_TRAINED, PlayerType.HUMAN, PlayerType.RANDOMAGENT, PlayerType.RANDOMAGENT]
	model_path = 'models/'

	mkgame = MachiKoroGame(player_types=player_types, trained_model_path=model_path)

	done = False
	while not done:
		done = mkgame.playtest_step(to_buy=mkgame.establishments_available_to_buy)

	print('game over')


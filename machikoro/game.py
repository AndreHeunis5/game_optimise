
import numpy as np
from machikoro.MachiKoroGame import MachiKoroGame
import time

verbose = False


if __name__ == "__main__":

	ngames = 10000
	win_count = {0: 0, 1: 0, 2: 0, 3: 0, }

	start = time.time()
	for ng in range(ngames):
		game = MachiKoroGame()

		nt = 0
		take_second_turn = False
		done = False
		while not done:
			if verbose: print('********* TURN ***********')
			nt += 1

			# state = game.build_state_vector()
			# TODO get action probs here
			action_probs = np.random.rand(game.ACTION_DIM)

			r, done, take_second_turn = game.step(action_probs=action_probs, take_second_turn=take_second_turn)

			# TODO rl agent training

		if verbose: print('after {} turns'.format(nt))
		for i, p in enumerate(game.players):
			win_count[i] += int(game._player_has_won(p))

	print(time.time() - start)
	print(win_count)







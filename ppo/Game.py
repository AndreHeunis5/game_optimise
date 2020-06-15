
from machikoro.MachiKoroGame import MachiKoroGame


class Game:

    def __init__(self):
        """
        ### Initialize
        """

        # create environment
        self.env = MachiKoroGame()

        # keep track of the episode rewards
        self.rewards = []

    def step(self, step_args):
        """
        ### Step
        Executes `action` for 4 time steps and
         returns a tuple of (observation, reward, done, take_second_turn, action_mask).
        * observation:
        * reward:           total reward while the action was executed
        * done:             whether the episode finished (a life lost)
        * take_second_turn
        * action_mask
        * episode_info
        """
        obs, r, done, take_second_turn, action_mask, game_length = self.env.step(step_args=step_args)

        obs = self.env.process_obs(obs)
        reward = r

        if done:
            # if finished, set episode information if episode is over, and reset
            episode_info = {"reward": reward, 'game_length': game_length}
            obs, action_mask = self.env.reset_game()
            take_second_turn = False
        else:
            episode_info = None

        return obs, reward, done, take_second_turn, action_mask, episode_info

    def reset(self):
        return self.env.reset_game()


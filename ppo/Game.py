
from machikoro.MachiKoroGame import MachiKoroGame


class Game:

    def __init__(self, player_types: list, trained_model_path: str, game_to_play: str):
        """
        ### Initialize
        """
        self.trained_model_path = trained_model_path

        # create environment
        if game_to_play == "machikoro":
            self.env = MachiKoroGame(player_types=player_types, trained_model_path=self.trained_model_path)
        else:
            raise ValueError('{} is not a supported game'.format(game_to_play))

        # keep track of the episode rewards
        self.rewards = []

    def step(self, step_args):
        """

        Executes `action` and returns a tuple of (observation, reward, done, take_second_turn, action_mask).
        * observation:
        * reward:           total reward while the action was executed
        * done:             whether the episode finished
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
            obs, action_mask = self.env.reset_game(self.trained_model_path)
            take_second_turn = False
        else:
            episode_info = None

        return obs, reward, done, take_second_turn, action_mask, episode_info

    def reset(self):
        return self.env.reset_game(self.trained_model_path)



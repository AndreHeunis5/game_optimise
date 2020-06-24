
import argparse
import torch
from pylab import *
import seaborn as sns

from ppo.Main import Main
from common.PlayerType import PlayerType

sns.set()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game',
                        help='Game to play',
                        type=str)
    parser.add_argument('--model_path',
                        help='Relative path to use for loading / storing trained model parameters',
                        type=str)
    parser.add_argument('--agent_update_freq',
                        help='Number of training steps between updating opponent agents / saving models',
                        type=int)
    parser.add_argument('--use_pretrained',
                        help='Whether to use a pretrained model or start training from scratch',
                        type=bool)
    args = parser.parse_args()

    model_path = args.model_path
    model_save_freq = args.agent_update_freq
    use_pretrained = args.use_pretrained
    game = args.game

    player_types = [
        PlayerType.AGENT_INTRAINING,
        PlayerType.AGENT_TRAINED,
        PlayerType.AGENT_TRAINED,
        PlayerType.AGENT_TRAINED]

    m = Main(
        game=game,
        device=device,
        player_types=player_types,
        model_save_freq=model_save_freq,
        start_from_pretrained=use_pretrained,
        pretrained_path=model_path)
    rewards, game_length = m.run_training_loop()
    m.destroy()

    torch.save(m.model.state_dict(), model_path + 'model.pt')

    figure(1)
    title('Average Rewards')
    plot(rewards)
    xlabel('training rounds')
    ylabel('average rewards')

    figure(2)
    title('Average Game Length')
    plot(game_length)
    xlabel('training rounds')
    ylabel('average game length')
    show()

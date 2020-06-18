
import torch
from ppo.Main import Main
from pylab import *
import seaborn as sns
sns.set()

from common.PlayerType import PlayerType

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if __name__ == "__main__":

    player_types = [
        PlayerType.AGENT_INTRAINING,
        PlayerType.AGENT_TRAINED,
        PlayerType.AGENT_TRAINED,
        PlayerType.AGENT_TRAINED]

    pretrained_path = '/Users/andreheunis/python_projects/game_opt/models/'
    m = Main(device=device, player_types=player_types, model_save_freq=120, start_from_pretrained=False, pretrained_path=pretrained_path)
    r, l = m.run_training_loop()
    m.destroy()

    torch.save(m.model.state_dict(), pretrained_path + 'model.pt')

    figure(1)
    title('Average Rewards')
    plot(r)
    xlabel('training rounds')
    ylabel('average rewards')

    figure(2)
    title('Average Game Length')
    plot(l)
    xlabel('training rounds')
    ylabel('average game length')
    show()

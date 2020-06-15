
import torch
from ppo.Main import Main
from pylab import *
import seaborn as sns
sns.set()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    m = Main(device=device)
    r, l = m.run_training_loop()
    m.destroy()

    torch.save(m.model.state_dict(), '../models/machikoro.pt')

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

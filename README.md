
# Game Optimise

The lofty goal is to have a tool that, given an implementation of a board / card game, can find optimal ways to play. 
This means that the tool could consist of 
* A PPO implementation that can become proficient at a given game
* An analysis tool to help derive any key features of the agent's play (TODO)

# Games

See directories of individual games for details of the implementation / findings.

* [Machikoro](machikoro): A city building card game

# Scripts
* **train_agent.py** Used for training an agent. Training starts off against agents that take random actions while the 
agent being trained improves. The weak agents are periodically updated with the main agent's policy.
* **playtest.py** Used to test how the trained agent does against a human opponent.



from enum import Enum


class PlayerType(Enum):
	HUMAN = 0
	AGENT_INTRAINING = 1
	AGENT_TRAINED = 2
	RANDOMAGENT = 3

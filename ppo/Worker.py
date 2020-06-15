
import multiprocessing
import multiprocessing.connection

from ppo.Game import Game


class Worker:
    """
    ## Worker
    Creates a new worker and runs it in a separate process.
    """
    child: multiprocessing.connection.Connection
    process: multiprocessing.Process

    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed))
        self.process.start()


def worker_process(remote: multiprocessing.connection.Connection, seed: int):
    """
    ##Worker Process
    Each worker process runs this method
    """

    # create game
    game = Game()

    # wait for instructions from the connection and execute them
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError("received {}".format(cmd))

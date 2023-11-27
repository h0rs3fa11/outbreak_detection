import random

def reward(nodes):
    # First round, there is no any placement
    if len(nodes) == 0:
        return 0
    # TODO:temporarily return a random number
    return random.randint(1, 50)

def cost(node):
    # TODO:temporarily return a random number
    if isinstance(node, list):
        return random.randint(1, 50)
    return random.randint(1, 50)
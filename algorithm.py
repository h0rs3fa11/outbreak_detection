import time
from outbreak import IC
from penalty_reduction import reward, cost
from queue import PriorityQueue

COST_TYPE = ['UC', 'CB']

def naive_greedy(G, B, cost_type='UC'):
    """
    G: graph
    B: budget
    type: unit code or variable cost

    Return: optimal set, time spending
    """
    # Initalization
    A = []
    max_reward = -1
    tmpG = G
    timelapse, start_time = [], time.time()

    while cost_type == 'UC' and len(A) < B or cost_type == 'CB' and cost(A) < B:
        current_set = A
        for node in tmpG.nodes():
            if cost_type == 'UC':
                r = reward(current_set.append(node)) - reward(A)
            elif cost_type == 'CB':
                r = (reward(current_set.append(node)) - reward(A)) / cost(node)
            else:
                raise ValueError(f'cost_type {cost_type} is not allowed')

            if r > max_reward:
                max_reward_node = node
        A.append(max_reward_node)
        tmpG.remove_node(max_reward_node)
        timelapse.append(time.time() - start_time)

    return (A, timelapse)


def marginal_gain(current_place, node, cost_type):
    r = reward(current_place.append(node)) - reward(current_place)
    return r if cost_type == 'UC' else r / cost(node)


def greedy_lazy_forward(G, B, cost_type='UC'):
    """
    G: graph
    B: budget
    type: unit code or variable cost

    Return: optimal set, time spending
    """
    # Initalization
    A = []
    timelapse, start_time = [], time.time()
    R = PriorityQueue(G.number_of_nodes())

    if cost_type not in COST_TYPE: 
        raise ValueError(f'cost_type {cost_type} is not allowed')
    
    # first round
    for node in G.nodes():
        R.put((marginal_gain(A, node, cost_type) * -1, node))
            
    while cost_type == 'UC' and len(A) < B or cost_type == 'CB' and cost(A) < B:
        top_node = R.get()

        # Re-evaluate the top node
        top_node[0] = marginal_gain(A, top_node[1], cost_type)

        # insert into the priority queue
        R.put(top_node)

        A.append(R.get()[1])

    timelapse.append(time.time() - start_time)

    return (A, timelapse)
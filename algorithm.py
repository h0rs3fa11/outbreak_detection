import time
from outbreak import IC
from marginal_gain import reward, cost


def naive_greedy_uc(G, B, cost_type='UC'):
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

    while cost_type == 'UC' and len(A) < B or cost_type == 'CB' and cost(A) < B :
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


def greedy_lazy_forward(G, B, cost_type='UC'):
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
    R = {}
    cur = {}

    for node in G.nodes():
        R[node] = float('inf')
    while cost_type == 'UC' and len(A) < B or cost_type == 'CB' and cost(A) < B :
        current_set = A
        
        for node in tmpG.nodes():
            cur[node] = False

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

def celf(g, k, p=0.1, mc=1000):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    # --------------------
    # Find the first node with greedy algorithm
    # --------------------

    # Calculate the first iteration sorted list
    start_time = time.time()
    marg_gain = [IC(g, [node], p, mc) for node in range(g.number_of_nodes())]

    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(range(g.number_of_nodes()), marg_gain),
               key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [
        g.number_of_nodes()], [time.time()-start_time]

    # --------------------
    # Find the next k-1 nodes using the list-sorting procedure
    # --------------------

    for _ in range(k-1):

        check, node_lookup = False, 0

        while not check:

            # Count the number of times the spread is computed
            node_lookup += 1

            # Recalculate spread of top node
            current = Q[0][0]

            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, IC(g, S+[current], p, mc) - spread)

            # Re-sort the list
            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        # Remove the selected node from the list
        Q = Q[1:]

    return (S, SPREAD, timelapse, LOOKUPS)

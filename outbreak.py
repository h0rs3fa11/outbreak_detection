from random import uniform, seed
import random
import numpy as np
import networkx as nx

def IC(g,S,p=0.5,mc=1000):
    """
    g: graph object
    S: set of seed nodes
    p: propagation probability
    mc: the number of Monte-Carlo simulations

    return: active nodes
    """

    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in range(mc):

        # Simulate propagation process
        new_active, A = S[:], S[:]
        while new_active:

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:

                # Determine neighbors that become infected
                np.random.seed(i)
                neighbors_list = list(g.neighbors(node))
                success = np.random.uniform(0,1,len(neighbors_list)) < p
                new_ones += list(np.extract(success, neighbors_list))

            new_active = list(set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A += new_active

        # The number of spreading 
        spread.append(len(A))

    return spread

def getOutbreaks(input_file_path, start_timestamp, end_timestamp):
    """
    For higgs dataset
    Read the timeline file, get the outbreaks(nodes, actions)
    """
    # Define the start and end timestamps
    # start_timestamp = 1341360000
    # end_timestamp = 1341446400

    output_file_path = 'filtered_activity_time.txt'

    # Read, filter, and write the data
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            parts = line.strip().split()
            if len(parts) == 4:
                timestamp = int(parts[2])
                if start_timestamp <= timestamp <= end_timestamp:
                    output_file.write(line)

    print("Filtered dataset created.")


# G = nx.gnm_random_graph(n=100, m=300, directed=True)

# seed_nodes = random.sample(list(G.nodes()), 3)
# IC(G, seed_nodes)
getOutbreaks('dataset/higgs-activity_time.txt', 1341360000, 1341446400)
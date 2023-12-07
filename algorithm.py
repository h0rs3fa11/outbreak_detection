import time
from queue import PriorityQueue
from network import Network
import networkx as nx
import logging
from tqdm import tqdm
import random

COST_TYPE = ['UC', 'CB']
OBJECTIVE_FUNCTION = ['DT', 'DL', 'PA']

class OutbreakDetection:
    def __init__(self, network, B, of, testing=True):
        logging.info('Initializing...')
        self.cost_types = ['UC', 'CB']
        if(not isinstance(network, Network)):
            raise TypeError('network must be a Network type')
        self.network = network
        self.G = self.network.G
        self.budget = B
        self.weakly_nodes, self.weakly_component = self.__init_weakly_component()
        # if testing:
        #     number_to_use = int(0.3 * len(self.weakly_component))
        #     sub_nodes = []
        #     sub_nodes = [n for i in range(number_to_use) for n in self.weakly_component[i]]
        #     self.G = nx.subgraph(self.G, sub_nodes)
        self.starting_points = self.__get_starting_point()

        # store the possibility(0 or 1) of each node then we don't have to calculate it in every iteration
        self.detection_likelihood_nodes = {}

        # objective function
        if(of not in OBJECTIVE_FUNCTION):
            raise ValueError('Objective function is not right')
        self.of = of

    def naive_greedy(self, cost_type):
        """
        type: unit code or variable cost

        Return: optimal set, time spending
        """
        # Initalization
        A = []
        # tmp_A = []
        total_cost = 0
        tmpG = self.G
        timelapse, start_time = [], time.time()

        logging.info('Running naive greedy algorithm...')
        while cost_type == 'UC' and len(A) < self.budget or cost_type == 'CB' and total_cost < self.budget:
            max_reward = 0
            logging.info(f'Current placement: {A}, cost type: {cost_type}, budget left: {self.budget - total_cost}')
            for node in tqdm(tmpG.nodes()):
                r = self.marginal_gain(A, node, cost_type)

                if r > max_reward:
                    max_reward_node = node
                    max_reward = r
            if(len(A) == 0): 
                logging.info('No node can benefit, exit')
                break
            A.append(max_reward_node)

            total_cost += self.network.node_cost[max_reward_node]

            tmpG.remove_node(max_reward_node)
            timelapse.append(time.time() - start_time)

        return (A, timelapse)


    def marginal_gain(self, current_place, node, cost_type):
        r = self.reward(current_place + [node]) - self.reward(current_place)
        return r if cost_type == 'UC' else r / self.network.node_cost[node]
    
    def reward(self, placement):
        """ Get reward of placement """
        if self.of == 'DL':
            total_reward = self.__detection_likelihood(placement)
        elif self.of == 'DT':
            total_reward = self.__detection_time(placement)
        else:
            total_reward = self.__population_affected(placement)
            
        return total_reward
    
    def __get_starting_point(self):
        starting_points = []
        components = self.weakly_component()
        for component in components:
            earlist_time = float('inf')
            starting_point = {}
            sub_graph = nx.subgraph(self.G, component)
            for u, v, d in sub_graph.edges(data=True):
                if(d['Timestamp'] < earlist_time):
                    # record the new edge
                    earlist_time = d['Timestamp']
                    starting_point['source'], starting_point['target'], starting_point['time'] = u, v, d['Timestamp']
            starting_points.append(starting_point)
        return starting_points

    def __detection_likelihood(self, placement):
        """
        Return: 0 or 1
        """
        # whether the node is in the same component with start point
        for n in placement:
            if n in self.detection_likelihood_nodes and self.detection_likelihood[n] == 1:
                return 1
            
            elif n not in self.detection_likelihood_nodes:
                for start in self.starting_points:
                    if not self.__in_same_weakly_component(start['source'], n):
                        continue
                    # get component id
                    component_id = self.weakly_nodes.get(start['source'])

                    # get component
                    sub_graph = nx.subgraph(self.G, self.weakly_component[component_id])
                    if nx.has_path(sub_graph, start['source'], n):
                        self.detection_likelihood_nodes[n] = 1
                        return 1
                self.detection_likelihood_nodes[n] = 0

        return 0

    def __detection_time():
        pass

    def __population_affected():
        pass

    def greedy_lazy_forward(self):
        """
        G: graph
        B: budget
        type: unit code or variable cost

        Return: optimal set, time spending
        """
        # Initalization
        A = []
        timelapse, start_time = [], time.time()
        R = PriorityQueue(self.G.number_of_nodes())

        if self.cost_type not in COST_TYPE: 
            raise ValueError(f'cost_type {self.cost_type} is not allowed')
        
        # first round
        for node in self.G.nodes():
            R.put((self.marginal_gain(A, node, self.cost_type) * -1, node))
                
        while self.cost_type == 'UC' and len(A) < self.budget or self.cost_type == 'CB' and self.cost(A) < self.budget:
            top_node = R.get()

            # Re-evaluate the top node
            top_node[0] = self.marginal_gain(A, top_node[1], self.cost_type)

            # insert into the priority queue
            R.put(top_node)

            A.append(R.get()[1])

        timelapse.append(time.time() - start_time)

        return (A, timelapse)
    
    def __get_weakly_component(self, threshold=10):
        filtered_components = list(filter(lambda x: len(x) > threshold, nx.weakly_connected_components(self.G)))
        return filtered_components
    
    # avoid to query weakly component frequently
    def __init_weakly_component(self, threshold=10):
        node_to_component = {}
        weakly_component = {}
        component_id = 0
        
        filtered = self.__get_weakly_component()
        for component in filtered:
            weakly_component[component_id] = component
            for node in component:
                node_to_component[node] = component_id
            component_id += 1
        return node_to_component, weakly_component
    
    def __in_same_weakly_component(self, node1, node2):
        return self.weakly_nodes.get(node1) == self.weakly_nodes.get(node2)
    
    def get_strongly_component(self):
        pass
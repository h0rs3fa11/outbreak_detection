import time
from queue import PriorityQueue
from network import Network
import networkx as nx
import logging
from tqdm import tqdm
import random
import math
import os
import csv
import pandas as pd

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
        if testing:
            test_file = 'dataset/test_subgraph.csv'
            if os.path.exists(test_file):
                df = pd.read_csv(test_file)
                # Create a directed graph
                self.G = nx.from_pandas_edgelist(df, source='Source', target='Target', edge_attr=True, create_using=nx.DiGraph())
            else:
                c = 0
                while c <= 5:
                    num_nodes_to_sample = int(0.3 * self.G.number_of_nodes())
                    sampled_nodes = random.sample(list(self.G.nodes()), num_nodes_to_sample)
                    subgraph = self.G.subgraph(sampled_nodes).copy()

                    filtered_components = list(filter(lambda x: len(x) > 10, nx.weakly_connected_components(subgraph)))
                    c = len(filtered_components)

                self.G = subgraph
                with open(test_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Source', 'Target', 'Timestamp', 'Follower_count'])

                    for u, v, data in self.G.edges(data=True):
                        writer.writerow([u, v, data['Timestamp'], data['Follower_count']])


        # self.weakly_nodes: {node: component_id}
        # self.weakly_component: {component_id: component node list}
        self.weakly_nodes, self.weakly_component = self.__init_weakly_component()

        # starting_points: [{'source': 1111, 'target': 2222, 'time': 12345}]
        self.starting_points = self.__get_starting_point()
        self.followers = self.__extract_followers()

        # Find time span
        self.t_max = self.__time_span()

        # store the possibility(0 to 1) of each node then we don't have to calculate it in every iteration
        self.detection_likelihood_nodes = {}
        # store the detection time of each node then we don't have to calculate it in every iteration
        self.detection_time_nodes = {}
        # objective function
        if(of not in OBJECTIVE_FUNCTION):
            raise ValueError('Objective function is not right')
        self.of = of

    def __time_span(self):
        largest_timestamp = float('-inf') 
        smallest_timestamp = float('inf') 

        for _, _, attrs in self.G.edges(data=True):
            timestamp = attrs.get('Timestamp', None)
            if timestamp is not None:
                largest_timestamp = max(largest_timestamp, timestamp)
                smallest_timestamp = min(smallest_timestamp, timestamp)

        # Check if any timestamp was found
        if largest_timestamp == float('-inf') or smallest_timestamp == float('inf'):
            raise ValueError('Couldn\'t find timestamp')
        else:
            return largest_timestamp - smallest_timestamp

    def naive_greedy(self, cost_type):
        """
        type: unit code or variable cost

        Return: optimal set, time spending
        """
        # Initalization
        A = []
        # tmp_A = []
        total_cost = 0
        tmpG = self.G.copy()
        timelapse, start_time = [], time.time()
        round_count = 1

        logging.info('Running naive greedy algorithm...')
        while total_cost < self.budget:
            max_reward = 0
            max_reward_node = None

            logging.info(f'Current placement size: {len(A)}, budget left: {self.budget - total_cost}')
            logging.debug(f'Current placement: {A}')

            for node in tqdm(tmpG.nodes()):
                r = self.marginal_gain(A, node, cost_type)

                if r > max_reward:
                    max_reward_node = node
                    max_reward = r

            if max_reward_node:
                A.append(max_reward_node)

            else: 
                logging.info('No node can benefit, exit')
                break
            logging.info(f'Finish the {round_count} round, the node with max reward is: {max_reward_node}, reward is {max_reward}')
            round_count += 1

            total_cost += self.network.node_cost[max_reward_node]

            if max_reward_node: tmpG.remove_node(max_reward_node)
            timelapse.append(time.time() - start_time)

        logging.info(f'The final placement has rewards {self.reward(A)}')
        return (A, timelapse[-1])

    def marginal_gain(self, current_place, node, cost_type):
        if current_place:
            r = self.reward(current_place + [node]) - self.reward(current_place)
        else:
            r = self.reward([node])
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
        components = self.weakly_component
        for c_id, component in components.items():
            earlist_time = float('inf')
            starting_point = {}
            sub_graph = nx.subgraph(self.G, component)
            for u, v, d in sub_graph.edges(data=True):
                if(d['Timestamp'] < earlist_time):
                    # record the new edge
                    earlist_time = d['Timestamp']
                    starting_point['source'], starting_point['target'], starting_point['time'] = u, v, d['Timestamp']
            starting_point['id'] = c_id
            starting_points.append(starting_point)
        return starting_points
    
    def __can_detect(self, placement):
        "Whether the placement can detect the outbreaks"
        # whether the node is in the same component with start point
        detected_outbreak_dl = set()

        # record the node that can detect the outbreak
        detected_info = {}
        for n in placement:
            # n can detect an outbreak
            if n in self.detection_likelihood_nodes and self.detection_likelihood_nodes[n] not in detected_outbreak_dl:
                # record the component id which n can detect
                detected_outbreak_dl.add(self.detection_likelihood_nodes[n])
                detected_info[n] = self.detection_likelihood_nodes[n]
            
            # new node, normally it is placement[-1]
            elif n not in self.detection_likelihood_nodes:
                # If n is not in the weakly connected components that we filtered, it won't detect the outbreak
                if n not in self.weakly_nodes:
                    continue
                # If n is in the same component with exist nodes(self.detection_likelihood_nodes), the reward won't increase
                if self.weakly_nodes.get(n) in detected_outbreak_dl:
                    # detected outbreak / all outbreaks
                    return detected_outbreak_dl, detected_info
                
                # For each cascade
                for start in self.starting_points:
                    if start['target'] == n:
                        continue
                    if not self.__in_same_weakly_component(start['target'], n):
                        continue
                    # get component id
                    component_id = start['id']

                    # get component
                    sub_graph = nx.subgraph(self.G, self.weakly_component[component_id])
                    if nx.has_path(sub_graph, start['target'], n):
                        self.detection_likelihood_nodes[n] = component_id
                        detected_outbreak_dl.add(component_id)
                        detected_info[n] = component_id
                        break
                    else:
                        # in this case, node n and this start node are in the same component, but have no paths, then we don't need to try the start point in another component
                        break
        return detected_outbreak_dl, detected_info
    
    def __detection_likelihood(self, placement):
        """
        Return: 0 to 1
        The fraction of all detected cascade
        """
        all_cascade = len(self.weakly_component)

        # detected_outbreak_dl is a list of component id which the current placement can detect
        detected_outbreak_dl, _ = self.__can_detect(placement)

        return len(detected_outbreak_dl) / all_cascade
    
    def __logarithmic_scaling(self, t):
        return 1 - math.log(t + 1) / math.log(self.t_max + 1)
    
    def __penalty_reduction_DT(self, T):
        if T == float('inf'):
            return 0
        else:
            return self.__logarithmic_scaling(T)

    def __detection_time(self, placement):
        """
        Return: 0 to 1
        """
        # detected_outbreak_dl is a list of component id which the current placement can detect
        _, detected_info = self.__can_detect(placement)
        if not detected_info: 
            return 0

        shortest_time = float('inf')
        # Get every starting point for each component
        for node in placement:
            if node not in detected_info:
                continue
            if node in self.detection_time_nodes:
                node_time = self.detection_time_nodes[node]
                shortest_time = node_time if shortest_time > node_time else shortest_time
                continue
            # Get the starting point of this component
            start_edge = self.starting_points[detected_info[node]]

            # How long it will take for the node to detect the information starts from start_edge
            shorted_path = nx.shortest_path(self.G, start_edge['target'], node)

            # directly connected with the starting point
            if len(shorted_path) < 2:
                shortest_time = 0
            else:
                time = self.G[shorted_path[-2]][shorted_path[-1]]['Timestamp'] - start_edge['time']
                if time < shortest_time:
                    shortest_time = time
            self.detection_time_nodes[node] = shortest_time
        if(shortest_time) < 0:
            raise ValueError(f'{shortest_time} cannot be negative')
        r = self.__penalty_reduction_DT(shortest_time)

        return r


    def __population_affected():
        pass

    def greedy_lazy_forward(self, cost_type):
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

        if cost_type not in COST_TYPE: 
            raise ValueError(f'cost_type {cost_type} is not allowed')
        
        # first round
        logging.info('First round, calculate all rewards')
        for node in tqdm(self.G.nodes()):
            R.put((self.marginal_gain(A, node, cost_type) * -1, node))
        
        pbar = tqdm(total=self.budget)

        logging.info('Start iterating...')

        costs = sum(self.network.node_cost[node] for node in A)
        while costs < self.budget:
            gain, top_node = R.get()

            # Re-evaluate the top node
            current_gain = self.marginal_gain(A, top_node, cost_type)
            logging.debug(f'Current gain for top node is {current_gain}')
            if current_gain == 0:
                break
            if current_gain == gain * -1:
                # If the top node's gain hasn't changed, add it to A
                A.append(top_node)
                pbar.update(costs - pbar.n)
            else:
                # Otherwise, reinsert it with the updated gain
                R.put((current_gain * -1, top_node))

            timelapse.append(time.time() - start_time)
        total_time = time.time() - start_time

        return (A, total_time)
    
    def celf(self):
        logging.info('CELF...')
        result_UC, time_UC = self.greedy_lazy_forward('UC')
        result_CB, time_CB = self.greedy_lazy_forward('CB')

        reward_UC = self.reward(result_UC)
        reward_CB = self.reward(result_CB)
        # time_UC.extend(time_CB)
        # time_log = sorted(time_UC)
        time_log = time_UC + time_CB

        logging.info(f'CELF: the reward result of UC is {reward_UC}, and CB is {reward_CB}')
        return result_UC, time_log if reward_UC > reward_CB else result_CB, time_log

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
        return self.weakly_nodes[node1] == self.weakly_nodes[node2]

    def get_bound(self, placement):
        raise Exception('not implemented')

    def heuristics(self, func_name):
        """
        Return: Placement A and total reward
        """
        total_cost = 0
        A = []
        left_nodes = list(self.G.nodes())

        while total_cost < self.budget:
            if func_name == 'random':
                new_node = self.__get_random_nodes(left_nodes)
            elif func_name == 'inlinks':
                new_node = self.__get_links_nodes(left_nodes, 'in')
            elif func_name == 'outlinks':
                new_node = self.__get_links_nodes(left_nodes, 'out')
            elif func_name == 'followers':
                new_node = self.__get_followers_count_nodes(left_nodes)
            else:
                raise ValueError(f'{func_name} not be supported')
            A.append(new_node)
            total_cost += self.network.node_cost[new_node]
        
        return self.reward(A)

    def __get_random_nodes(self, nodes):
        return random.choice(nodes)
    
    def __get_links_nodes(self, nodes, d):
        max_degree = [0, -1]
        for n in nodes:
            if d == 'in':
                degree = self.G.in_degree(n)
            elif d == 'out':
                degree = self.G.out_degree(n)
            else:
                raise ValueError(f'degree type {d} is invalid')
            if degree > max_degree[1]:
                max_degree = [n, degree]
        return max_degree[0]

    def __get_followers_count_nodes(self, nodes):
        max_follower = [0, -1]
        for n in nodes:
            follower_count = 0 if n not in self.followers else self.followers[n]
            if follower_count > max_follower[1]:
                max_follower = [n, follower_count]
        return max_follower[0]

    def __extract_followers(self):
        follower_map ={}
        for _, v, d in self.G.edges(data=True):
            follower_map[v] = d['Follower_count']
        return follower_map
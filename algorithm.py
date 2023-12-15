import time
from queue import PriorityQueue
from network import Network
import networkx as nx
import logging
from tqdm import tqdm
import random
import json
import os
import csv
import pandas as pd
from utils.utils import intersect_all_sets, output, normalize_numbers

COST_TYPE = ['UC', 'CB']
OBJECTIVE_FUNCTION = ['DT', 'DL', 'PA']

class OutbreakDetection:
    def __init__(self, network, B, of, testing=True, use_follower=False):
        logging.info('Initializing...')
        self.cost_types = ['UC', 'CB']
        if(not isinstance(network, Network)):
            raise TypeError('network must be a Network type')
        self.network = network
        self.G = self.network.G
        self.budget = B
        self.use_follower = use_follower
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
                    writer.writerow(['Source', 'Target', 'Timestamp'])

                    for u, v, data in self.G.edges(data=True):
                        writer.writerow([u, v, data['Timestamp']])


        # self.weakly_nodes: {node: component_id}
        # self.weakly_component: {component_id: component node list}
        self.weakly_nodes, self.weakly_component = self.__init_weakly_component()

        # starting_points: [{'source': 1111, 'target': 2222, 'time': 12345}]
        self.starting_points = self.__get_starting_point()
        if use_follower:
            self.followers = self.__extract_followers()

        # Find time span
        self.t_max = self.__time_span()

        # store the possibility(0 to 1) of each node then we don't have to calculate it in every iteration
        self.detection_likelihood_nodes = {}
        # store the detection time of each node then we don't have to calculate it in every iteration
        self.detection_time_nodes = {}
        # store the population affected of each node then we don't have to calculate it in every iteration
        self.population_affected = {}
        self.reward_cache = {}
        # objective function
        if(of not in OBJECTIVE_FUNCTION):
            raise ValueError('Objective function is not right')
        self.of = of

        ic_path = f'{self.network.dataset_dir}/ic_{self.network.activity}.json'
        if self.of == 'PA':
            if not os.path.exists(ic_path):
                logging.info('Extracting information cascades...')
                self.cascades = self.__information_cascade()
                with open(ic_path, 'w') as f:
                    json.dump(self.cascades, f)
            else:
                with open(ic_path, 'r') as f:
                    self.cascades = json.loads(f.read())
            self.cascades = {int(key): value for key, value in self.cascades.items()}
            self.followers_affected = 0
            if self.use_follower:
                for n in self.cascades:
                    if n in self.followers:
                        self.followers_affected += len(self.followers[n])
            self.all_affected = {}
            for cid in self.weakly_component:
                self.all_affected[cid] = []

            for n in self.cascades:
                self.all_affected[self.weakly_nodes[n]].append(n)

    def __time_span(self):
        max_times = {}
        for cid, nodes in self.weakly_component.items():
            largest_timestamp = float('-inf') 
            smallest_timestamp = float('inf') 

            sub_graph = nx.subgraph(self.G, nodes)
            for _, _, attrs in sub_graph.edges(data=True):
                timestamp = attrs.get('Timestamp', None)
                if timestamp is not None:
                    largest_timestamp = max(largest_timestamp, timestamp)
                    smallest_timestamp = min(smallest_timestamp, timestamp)

            # Check if any timestamp was found
            if largest_timestamp == float('-inf') or smallest_timestamp == float('inf'):
                raise ValueError('Couldn\'t find timestamp')
            else:
                max_times[cid] = largest_timestamp - smallest_timestamp
        return max_times
    
    def __remove_startpoint_from_p(self, G):
        for item in self.starting_points.values():
            G.remove_node(item['source'])
            G.remove_node(item['target'])

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
        start_time = time.time()
        time_logs = {}
        round_count = 1

        self.__remove_startpoint_from_p(tmpG)

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
                logging.debug(f'Found {max_reward_node} with {max_reward} marginal benefit')
                A.append(max_reward_node)
                cur_reward = self.reward(A)
                time_logs[len(A)] = {'runtime':time.time() - start_time, 'reward': cur_reward}
                logging.info(f'Selected {len(A)} nodes, with reward {cur_reward}, spent {time_logs[len(A)]}')

            else: 
                logging.info('No node can benefit, exit')
                break
            logging.info(f'Finish the {round_count} round, the node with max reward is: {max_reward_node}, (marginal)reward is {max_reward}')
            round_count += 1

            total_cost += self.network.node_cost[max_reward_node]

            if max_reward_node: tmpG.remove_node(max_reward_node)

        if total_cost >= self.budget:
            logging.debug('Budget is exhausted')
        final_reward = self.reward(A)
        logging.info(f'The final placement has rewards {final_reward}')
        return output(f'greedy-{cost_type}', A, final_reward, time_logs)

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
        starting_points = {}
        components = self.weakly_component
        for c_id, component in components.items():
            earlist_time = float('inf')
            starting_point = {}
            sub_graph = nx.subgraph(self.G, component)
            for u, v, d in sub_graph.edges(data=True):
                # self loop
                if u == v:
                    continue
                if(d['Timestamp'] < earlist_time):
                    # record the new edge
                    earlist_time = d['Timestamp']
                    starting_point['source'], starting_point['target'], starting_point['time'] = u, v, d['Timestamp']
            starting_points[c_id] = starting_point
        return starting_points
    
    def __can_detect(self, placement):
        "Whether the placement can detect the outbreaks"
        # whether the node is in the same component with start point
        detected_outbreak_dl = set()

        # record the node that can detect the outbreak
        detected_info = {}
        for n in placement:
            # n can detect an outbreak
            if n in self.detection_likelihood_nodes:
                # record the component id which n can detect
                detected_outbreak_dl.add(self.detection_likelihood_nodes[n])
                detected_info[n] = self.detection_likelihood_nodes[n]
            
            # new node, normally it is placement[-1]
            elif n not in self.detection_likelihood_nodes:
                # If n is not in the weakly connected components that we filtered, it won't detect the outbreak
                if n not in self.weakly_nodes:
                    continue
                cid = self.weakly_nodes.get(n)
                # If n is in the same component with exist nodes(self.detection_likelihood_nodes), the reward won't increase
                if cid in detected_outbreak_dl:
                    # detected outbreak / all outbreaks
                    detected_info[n] = cid
                    return detected_outbreak_dl, detected_info
                
                # For the cascade of this component
                start = self.starting_points[cid]
                # cannot be the starting point itself
                if start['target'] == n or start['source'] == n:
                    continue
                # get component
                sub_graph = nx.subgraph(self.G, self.weakly_component[cid])
                if nx.has_path(sub_graph, start['target'], n) or nx.has_path(sub_graph, start['source'], n):
                    self.detection_likelihood_nodes[n] = cid
                    detected_outbreak_dl.add(cid)
                    detected_info[n] = cid
                    # break
                
        # if detected_outbreak_dl:
        #     logging.debug(f'\n{placement} can detect outbreak in components {detected_outbreak_dl}')

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
    
    def __penalty_reduction_DT(self, T):
        sum_frac = 0
        for cid, t in T.items():
            if t == float('inf'):
                continue
            sum_frac += (1 - t / self.t_max[cid])
        return sum_frac
    
    def __detection_time(self, placement):
        """
        Return: 0 to 1
        """
        # detected_outbreak_dl is a list of component id which the current placement can detect
        detected_components, detected_info = self.__can_detect(placement)
        if not detected_info: 
            return 0

        shortest_time = {key: float('inf') for key in self.weakly_component}
        # Get every starting point for each component
        for node in placement:
            if node not in detected_info:
                continue
            component_id = self.weakly_nodes[node]
            if node in self.detection_time_nodes:
                node_time = self.detection_time_nodes[node]
                shortest_time[component_id] = node_time if shortest_time[component_id] > node_time else shortest_time[component_id]
                continue
            # Get the starting point of this component
            start_edge = self.starting_points[detected_info[node]]

            # How long it will take for the node to detect the information starts from start_edge
            if nx.has_path(self.G, start_edge['target'], node) and nx.has_path(self.G, start_edge['source'], node):
                shorted_path_a = nx.shortest_path(self.G, start_edge['target'], node)
                shorted_path_b = nx.shortest_path(self.G, start_edge['source'], node)
                time = min(self.G[shorted_path_a[-2]][shorted_path_a[-1]]['Timestamp'], self.G[shorted_path_b[-2]][shorted_path_b[-1]]['Timestamp']) - start_edge['time']
            elif nx.has_path(self.G, start_edge['target'], node):
                shorted_path = nx.shortest_path(self.G, start_edge['target'], node)
                time = self.G[shorted_path[-2]][shorted_path[-1]]['Timestamp'] - start_edge['time']
            else:
                shorted_path = nx.shortest_path(self.G, start_edge['source'], node)
                time = self.G[shorted_path[-2]][shorted_path[-1]]['Timestamp'] - start_edge['time']

            shortest_time[component_id] = time if time < shortest_time[component_id] else shortest_time[component_id]
            self.detection_time_nodes[node] = shortest_time[component_id]

            # cache
            self.reward_cache[node] = self.__penalty_reduction_DT({component_id: self.detection_time_nodes[node]})

        if(shortest_time[component_id]) < 0:
            raise ValueError(f'{shortest_time} cannot be negative')
        
        r = self.__penalty_reduction_DT(shortest_time)

        return r

    def __information_cascade(self):
        """
        Record the information cascade details
        {
          "node_id": "min_activity_time_from_the_start_point"
        }
        """

        cascades = {}

        for component_id, node_list in tqdm(self.weakly_component.items()):
            start = self.starting_points[component_id]
            for node in tqdm(node_list):
                if node == start['source'] or node == start['target']:
                    continue
                if not (nx.has_path(self.G, start['target'], node) or nx.has_path(self.G, start['source'], node)):
                    continue
                if nx.has_path(self.G, start['target'], node) and nx.has_path(self.G, start['source'], node):
                    start_point = (start['source'], start['target'])
    
                elif not nx.has_path(self.G, start['target'], node):
                    start_point = [start['source']]
                else:
                    start_point = [start['target']]

                min_time = float('inf')
                for s in start_point:
                        shortest_path = nx.dijkstra_path(self.G, s, node, 'Timestamp')
                        if len(shortest_path) == 1:
                            timestamp = self.G[s][shortest_path[0]]['Timestamp']
                        else:
                            timestamp = self.G[shortest_path[-2]][shortest_path[-1]]['Timestamp']
                        min_time = min(min_time, timestamp)
                    
                cascades[node] = min_time

        return cascades
    
    def __population_affected(self, placement):
        """
        Return: fraction of all nodes in that component
        """
        # TODO:区别计算不同的component
        affected_of_placement, affected_node, benefit = {}, {}, {}
        for cid in self.weakly_component:
            affected_of_placement[cid], affected_node[cid] = [], set()

        for node in placement:
            affected = set()
            if node not in self.cascades:
                # cannot detect anything
                for cid in self.weakly_component:
                    affected_of_placement[cid].append(self.all_affected[cid])
                    if node not in self.population_affected:
                        self.population_affected[node] = self.all_affected[cid]
                continue
            component_id = self.weakly_nodes[node]
            if node == self.starting_points[component_id]['source'] or node ==self.starting_points[component_id]['target']:
                continue  
            detect_time_at = self.cascades[node]

            if node in self.population_affected:
                aff = self.population_affected[node]
                for a in aff:
                    affected.add(a)
            else:
                self.population_affected[node] = set()
                for n in self.cascades:
                    # whether the cascade nodes and sensor are in the same component
                    if component_id != self.weakly_nodes[n]:
                        # nodes in other component will be affected
                        affected.add(n)
                        self.population_affected[node].add(n)
                    # check whether node n is affected before detect_time_at
                    elif self.cascades[n] < detect_time_at:
                        affected.add(n)
                        self.population_affected[node].add(n)

            affected_of_placement[component_id].append(affected)

        for cid in self.weakly_component:
            # get the intersection of affected group of each selected node
            affected_node[cid] = intersect_all_sets(affected_of_placement[cid])

            # if len(affected_node[cid]) == len(self.all_affected[cid]):
            #     benefit[cid] = 0
        
        # follower_aff = set()
        # if self.use_follower: # not finish
        #     follower_aff = set().union(*(self.followers[n] for n in affected_node if n in self.followers))

        # benefit[component_id] = 1 - (len(affected_node[component_id]) + len(follower_aff)) / (self.all_affected[component_id] + self.followers_affected)
            if len(self.all_affected[cid]) == 0:
                benefit[cid] = 0
            else:
                benefit[cid] = 1 - len(affected_node[cid]) / len(self.all_affected[cid])

        # skip the node that can make reward of 1
        return sum([x for x in benefit.values()])

    def greedy_lazy_forward(self, cost_type):
        """
        type: unit code or variable cost

        Return: optimal set, time spending
        """
        # Initalization
        A = []
        timelapse, start_time = {}, time.time()
        tmpG = self.G.copy()
        self.__remove_startpoint_from_p(tmpG)

        R = PriorityQueue(tmpG.number_of_nodes())

        if cost_type not in COST_TYPE: 
            raise ValueError(f'cost_type {cost_type} is not allowed')
        
        # first round
        logging.info('First round, calculate all rewards')
        for node in tqdm(tmpG.nodes()):
            R.put((self.marginal_gain(A, node, cost_type) * -1, node))
        
        A.append(R.get()[1])
        cur_reward = self.reward(A)
        timelapse[len(A)] = {'runtime':time.time() - start_time, 'reward': cur_reward}
        logging.info(f'Selected {len(A)} nodes, with reward {cur_reward}, spent {timelapse[len(A)]}')
        pbar = tqdm(total=self.budget)

        logging.info('\Start iterating...')

        costs = sum(self.network.node_cost[node] for node in A)
        while costs < self.budget:
            gain, top_node = R.get()
            if gain == 0:
                logging.debug('No node can benefit, exit')
                break
            # Re-evaluate the top node
            current_gain = self.marginal_gain(A, top_node, cost_type)
            # logging.debug(f'Current gain for top node is {current_gain}')

            if current_gain == gain * -1:
                # If the top node's gain hasn't changed, add it to A
                A.append(top_node)
                cur_reward = self.reward(A)
                timelapse[len(A)] = {'runtime':time.time() - start_time, 'reward': cur_reward}
                logging.info(f'Selected {len(A)} nodes, with reward {cur_reward}, spent {timelapse[len(A)]}')
                pbar.update(costs - pbar.n)
            else:
                # Otherwise, reinsert it with the updated gain
                R.put((current_gain * -1, top_node))

            costs = sum(self.network.node_cost[node] for node in A)

        if costs >= self.budget:
            logging.debug('Budget is exhausted')

        return (A, timelapse)
    
    def celf(self):
        """Returns
        {
            'algo': 'celf',
            'placement': BEST_PLACEMENT,
            'reward': xxx,
            'runtime': xxx(s)
        }
        """
        logging.info('CELF...')
        result = {}
        result['algo'] = 'celf'
        result_UC, time_UC = self.greedy_lazy_forward('UC')
        result_CB, time_CB = self.greedy_lazy_forward('CB')

        reward_UC = self.reward(result_UC)
        reward_CB = self.reward(result_CB)
        # time_UC.extend(time_CB)
        # time_log = sorted(time_UC)
        time_log = {}
        if reward_UC > reward_CB:
            time_log = time_UC
        else:
            time_log = time_CB

        logging.info(f'CELF: the reward result of UC is {reward_UC}, and CB is {reward_CB}')
        logging.debug(f'CELF: the placement of UC is {result_UC}, and CB is {result_CB}')

        return output('celf', result_UC, reward_UC, time_log) if reward_UC > reward_CB else output('celf', result_CB, reward_CB, time_log)

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

    def get_bound(self, placement):
        raise Exception('not implemented')

    def heuristics(self, func_name):
        """
        Return: Placement A and total reward
        """
        total_cost = 0
        A = []
        reward = {}
        tmpG = self.G.copy()
        pbar = tqdm(total=self.budget)

        while total_cost < self.budget:
            if func_name == 'random':
                new_node = self.__get_random_nodes(list(tmpG.nodes()))
            elif func_name == 'inlinks':
                new_node = self.__get_links_nodes(list(tmpG.nodes()), 'in')
            elif func_name == 'outlinks':
                new_node = self.__get_links_nodes(list(tmpG.nodes()), 'out')
            elif func_name == 'followers':
                if self.use_follower:
                    new_node = self.__get_followers_count_nodes(list(tmpG.nodes()))
                else:
                    raise ValueError('Follower function haven\'t activated')
            else:
                raise ValueError(f'{func_name} not be supported')
            if new_node in self.reward_cache:
                node_reward = self.reward_cache[new_node]
            else:
                node_reward = self.reward([new_node])
            if node_reward == 0:
                continue
            A.append(new_node)
            reward[len(A)] = self.reward(A)
            total_cost += self.network.node_cost[new_node]
            pbar.update(total_cost - pbar.n)

            tmpG.remove_node(new_node)
        
        return {'algo': func_name, 'placement': A, 'reward': reward}

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
            follower_count = 0 if n not in self.followers else len(self.followers[n])
            if follower_count > max_follower[1]:
                max_follower = [n, follower_count]
        return max_follower[0]

    def __extract_followers(self):
        with open(self.network.follower_path, 'r') as f:
            data = json.load(f)
        
        follower_map = {int(key): value for key, value in data.items()}
        
        return follower_map
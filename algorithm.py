import time
from queue import PriorityQueue
from network import Network

COST_TYPE = ['UC', 'CB']

class OutbreakDetection:
    def __init__(self, network, B):
        self.cost_types = ['UC', 'CB']
        if(not isinstance(network, Network)):
            raise TypeError('network must be a Network type')
        self.network = network
        self.G = self.network.G
        self.budget = B

    def naive_greedy(self, cost_type):
        """
        G: Netowrk graph
        B: budget
        type: unit code or variable cost

        Return: optimal set, time spending
        """
        # Initalization
        A = []
        
        total_cost = 0
        tmpG = self.G
        timelapse, start_time = [], time.time()

        while cost_type == 'UC' and len(A) < self.budget or cost_type == 'CB' and total_cost < self.budget:
            max_reward = -1
            for node in tmpG.nodes():
                r = self.marginal_gain(A, node, cost_type)

                if r > max_reward:
                    max_reward_node = node
                    max_reward = r
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
        """ TODO: This is a temporary version of calculating reward, it simply returns a value that is less than previous one, to satisfy the submodularity property """
        base_reward = 100
        total_reward = 0
        if not placement: return 0
        # Reward calculation logic
        for i, node in enumerate(placement):
            # Decrease reward incrementally for each subsequent node
            total_reward += base_reward * (node / 1000) / (i + 1)
        return total_reward

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
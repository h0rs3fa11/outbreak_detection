from network import Network
from algorithm import OutbreakDetection
import logging
# initialization, data pre-processing

logging.basicConfig(level=logging.INFO)

mt_n = Network(
    dataset_dir='dataset', 
    original_file_name='higgs-mention_network.edgelist', 
    result_file_name='mention_higgs_network.csv', 
    follower_file_name='higgs-social_network.edgelist', 
    timestamp_file_name='higgs-activity_time.txt',
    activity='MT', init=True)

rt_n = Network(
    dataset_dir='dataset',
    original_file_name='higgs-retweet_network.edgelist',
    result_file_name='retweet_higgs_network.csv',
    follower_file_name='higgs-social_network.edgelist',
    timestamp_file_name='higgs-activity_time.txt',
    activity='RT', init=True)

mt_n.simulate_cost(rt_n)

# run algorithm
algo = OutbreakDetection(mt_n, 1000, 'DL')
print(algo.naive_greedy('UC'))
print(algo.naive_greedy('CB'))
print(algo.celf())

# solution quality

# creata result graphs
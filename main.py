from network import Network
# initialization, data pre-processing

mt_n = Network(
    dataset_dir='dataset', 
    original_file_name='higgs-mention_network.edgelist', 
    result_file_name='mention_higgs_network.csv', 
    follower_file_name='higgs-social_network.edgelist', 
    timestamp_file_name='higgs-activity_time.txt',
    activity='MT')

rt_n = Network(
    dataset_dir='dataset',
    original_file_name='higgs-retweet_network.edgelist',
    result_file_name='retweet_higgs_network.csv',
    follower_file_name='higgs-social_network.edgelist',
    timestamp_file_name='higgs-activity_time.txt',
    activity='RT')

mt_n.simulate_cost(rt_n)

# run algorithm

# solution quality

# creata result graphs
# Outbreak Detection

Outbreak detection by using CELF Algorithm

## Datasets

Put [higgs dataset files](https://snap.stanford.edu/data/higgs-twitter.html) in `dataset/`

## Usage

Initialize the network

```python
from network import Network

# network of mentions
mt_n = Network(
    dataset_dir='dataset', 
    original_file_name='higgs-mention_network.edgelist', 
    result_file_name='mention_higgs_network.csv', 
    follower_file_name='higgs-social_network.edgelist', 
    timestamp_file_name='higgs-activity_time.txt',
    activity='MT')

# network of retweets
rt_n = Network(
    dataset_dir='dataset',
    original_file_name='higgs-retweet_network.edgelist',
    result_file_name='retweet_higgs_network.csv',
    follower_file_name='higgs-social_network.edgelist',
    timestamp_file_name='higgs-activity_time.txt',
    activity='RT')


# Record all nodes and its simulated cost
mt_n.simulate_cost(rt_n)

# mt_n.node_cost['NODE_NAME']
```

## TODOs

### Dataset preprocessing

- [x] Count the number of followers for nodes in the retweet and mention graph.

- [x] Simulate the cost of monitoring each node, create a new graph to indicate the cost

- [x] Extract information cascade

- [ ] Define a function to calculate penalty reduction

  - [x] detection time
  - [x] detection likelihood
  - [ ] population affected

- [x] Greedy/CELF algorithm

- [x] heuristic approaches

- [ ] Solution quality

- [ ] Testing

- [ ] Result graph

- [ ] configparser(optional)

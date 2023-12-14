from network import Network
from algorithm import OutbreakDetection
import logging
import argparse
import random
import json
import os

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Find the best placement for outbreak detection')
parser.add_argument('-test', '--test', type=int, choices=[1, 0], default=1, help='Whether to use a smaller dataset for testing')
parser.add_argument('-g', '--graph', required=True, choices=['mt', 'rt'], help='indicate which graph to use, mt(mention) or rt(retweet)')
parser.add_argument('-b', '--budget', required=True, help='total budget', type=int)
parser.add_argument('-obj', '--objective_function', required=True, choices=['DL', 'DT', 'PA'], help='objective functions')

sub_parsers = parser.add_subparsers(dest='command', required=True, help='sub command help')

subparser = sub_parsers.add_parser('algorithm', help='algorithm: choose algorithms and objective functions')
subparser.add_argument('-algo', '--algorithm', required=True, choices=['uc-greedy', 'cb-greedy', 'celf'], help='algorithm to find the placement for outbreak detection\n uc-greedy: unit cost case, cb-greedy: variable cost case, celf: CELF algorithm')

subparser_heuristic = sub_parsers.add_parser('heuristic', help='heuristic: use heuristic methods to find placement')
subparser_heuristic.add_argument('-f', '--function', required=True, choices=['inlinks', 'outlinks', 'random', 'followers'])

args = parser.parse_args()

if args.test:
    random_seed = 1 
    random.seed(random_seed)

if args.graph == 'mt':
    network = Network(
        dataset_dir='dataset', 
        original_file_name='higgs-mention_network.edgelist', 
        result_file_name='mention_higgs_network.csv', 
        follower_file_name='higgs-social_network.edgelist', 
        timestamp_file_name='higgs-activity_time.txt',
        activity='MT')
else:
    network = Network(
        dataset_dir='dataset',
        original_file_name='higgs-retweet_network.edgelist',
        result_file_name='retweet_higgs_network.csv',
        follower_file_name='higgs-social_network.edgelist',
        timestamp_file_name='higgs-activity_time.txt',
        activity='RT')

# run algorithm
algo = OutbreakDetection(network, args.budget, args.objective_function, testing=args.test)

if args.command == 'algorithm':
    if args.algorithm == 'uc-greedy':
        result = algo.naive_greedy('UC')
    elif args.algorithm == 'cb-greedy':
        result = algo.naive_greedy('CB')
    elif args.algorithm == 'celf':
        result = algo.celf()

elif args.command == 'heuristic':
    result = algo.heuristics(args.function)

logging.info(result)

result_path = f'results/{args.graph}-result-{args.objective_function}.json'
results = []

# record the result
if os.path.exists(result_path):
    with open(result_path, 'r') as f:
        try:
            results = json.loads(f.read())
        except json.JSONDecodeError:
            results = []
results.append(result)
with open(result_path, 'w') as f:
    json.dump(results, f)

# creata result graphs
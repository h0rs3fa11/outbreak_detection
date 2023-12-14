import pandas as pd
import networkx as nx
import random
import os
import json
import logging

class Network:
    def __init__(self, dataset_dir, original_file_name, result_file_name, follower_file_name, timestamp_file_name, activity):
        self.activities = ['MT', 'RT']
        if(activity not in self.activities):
            raise Exception('Invalid activity type')
        self.activity = activity
        self.dataset_dir = dataset_dir
        self.node_cost = {}
        self.graph =  f'{self.dataset_dir}/{result_file_name}'
        # retweet or mention
        self.file_path = f'{self.dataset_dir}/{original_file_name}'

        self.file_path_follower = f'{self.dataset_dir}/{follower_file_name}'

        self.timestamp_path = f'{self.dataset_dir}/{timestamp_file_name}'

        if not os.path.exists(self.graph):
            logging.info('Loading the original datasets...')
            self.pre_processing(activity)
        
        self.G = self.load_network()

        self.cost_path = f'{self.dataset_dir}/cost-{activity}.json'
        self.follower_path = f'{self.dataset_dir}/followers-{activity}.json'

        if not os.path.exists(self.follower_path):
            logging.info('Loading follower information...')
            fl_edge_list = Network.read_network(self.file_path_follower, names=['Follower', 'User'])

            follower_dict = fl_edge_list.groupby('User')['Follower'].apply(list).to_dict()
            with open(self.follower_path, 'w') as f:
                json.dump(follower_dict, f)


        if not os.path.exists(self.cost_path):
            logging.info('Generating cost information...')
            self.simulate_cost(self.cost_path)
        else:
            with open(self.cost_path, 'r') as f:
                data = json.loads(f.read())
            self.node_cost = {int(key): value for key, value in data.items()}

    @staticmethod
    def read_network(file_path, names=["Source", "Target", "Weight"]):
        edge_list_df = pd.read_csv(file_path, sep=" ", header=None, names=names)
        return edge_list_df

    def pre_processing(self, activity):
        """ Process the original dataset to csv files """
        edge_list = Network.read_network(self.file_path)
        timestamp_df = Network.read_network(self.timestamp_path, names=["Source", "Target", "Timestamp", "Activity"])

        edge_list['Activity'] = activity

        # include timestamp
        merged_df = pd.merge(edge_list, timestamp_df,  how='left', left_on=['Source','Target','Activity'], right_on = ['Source','Target', 'Activity'])

        merged_df.drop(['Activity', 'Weight'], axis=1, inplace=True)

        # reverse the direction to match the information spreading flow
        merged_df['Source'], merged_df['Target'] = merged_df['Target'], merged_df['Source']


        csv_file = self.graph

        merged_df.to_csv(csv_file, index=False)

    def load_network(self):
        """Load network from csv file"""
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.graph)

        # Create a directed graph
        G = nx.from_pandas_edgelist(df, source='Source', target='Target', edge_attr=True, create_using=nx.DiGraph())

        return G

    def simulate_cost(self, filename):
        nodes = set(self.G.nodes())
        for n in nodes:
            self.node_cost[n] = random.randint(1, 1000)
        with open(filename, 'w') as f:
            json.dump(self.node_cost, f)

import pandas as pd
import networkx as nx
import random

class Network:
    def __init__(self, dataset_dir, original_file_name, result_file_name, follower_file_name, timestamp_file_name, activity, init=False, testing=True):
        self.activities = ['MT', 'RT']
        if(activity not in self.activities):
            raise Exception('Invalid activity type')
        
        self.dataset_dir = dataset_dir
        self.node_cost = {}
        self.graph = self.dataset_dir + '/' + result_file_name
        self.testing = testing
        # retweet or mention
        self.file_path = self.dataset_dir + '/' + original_file_name

        self.file_path_follower = self.dataset_dir + '/' + follower_file_name

        self.timestamp_path = self.dataset_dir + '/' + timestamp_file_name

        if not init:
            self.pre_processing(activity)
        
        self.G = self.load_network()

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

        fl_edge_list = Network.read_network(self.file_path_follower, names=['Follower', 'User'])

        followers_count = fl_edge_list.groupby('User').size()
        follower_dict = followers_count.to_dict()

        merged_df['Follower_count'] = merged_df['Target'].map(follower_dict)
        merged_df['Follower_count'].fillna(0, inplace=True)

        merged_df['Follower_count'] = merged_df['Follower_count'].astype(int)

        csv_file = self.graph

        merged_df.to_csv(csv_file, index=False)

    def load_network(self):
        """Load network from csv file"""
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.graph)

        # Create a directed graph
        G = nx.from_pandas_edgelist(df, source='Source', target='Target', edge_attr=True, create_using=nx.DiGraph())

        return G

    def simulate_cost(self, n2):
        if(not isinstance(n2, Network)):
            raise Exception('Wrong n2 type')
        
        nodes = set(self.G.nodes()).union(set(n2.G.nodes()))
        for n in nodes:
            self.node_cost[n] = random.randint(1, 1000)

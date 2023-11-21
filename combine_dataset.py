"""
Combine four files of higgs data
"""
import pandas as pd

def read_network(file_path, names=["Source", "Target", "Weight"]):
    edge_list_df = pd.read_csv(file_path, sep=" ", header=None, names=names)
    return edge_list_df

file_path_mention = 'dataset/higgs-mention_network.edgelist'
file_path_reply = 'dataset/higgs-reply_network.edgelist'
file_path_retweet = 'dataset/higgs-retweet_network.edgelist'
timestamp_path = "dataset/higgs-activity_time.txt"

edge_list_mt = read_network(file_path_mention)
edge_list_re = read_network(file_path_reply)
edge_list_rt = read_network(file_path_retweet)
timestamp_df = read_network(timestamp_path, names=["Source", "Target", "Timestamp", "Activity"])

edge_list_mt["Activity"] = "MT"
edge_list_re["Activity"] = "RE"
edge_list_rt["Activity"] = "RT"

# include timestamp
combined_df = pd.concat([edge_list_mt, edge_list_re, edge_list_rt])

merged_df = pd.merge(combined_df, timestamp_df,  how='left', left_on=['Source','Target','Activity'], right_on = ['Source','Target', 'Activity'])
# 329379 13813
# merged_df.drop('Activity', axis=1, inplace=True)

csv_file = "dataset/combined_higgs_network.csv"

merged_df.to_csv(csv_file, index=False)
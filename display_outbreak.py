"""
Display the trending of this dataset
"""
import pandas as pd
import matplotlib.pyplot as plt

# Function to safely read each line
def read_line(line):
    try:
        user1, user2, timestamp, interaction = line.split()
        return {'user1': user1, 'user2': user2, 'timestamp': int(timestamp), 'interaction': interaction}
    except ValueError:
        print(line)
        return None  # or handle the error as you see fit

# Read and parse the file
file_path = 'dataset/higgs-activity_time.txt'  # Replace with your file path
with open(file_path, 'r') as file:
    lines = file.readlines()

# Convert lines to dictionary and then to DataFrame
data = [read_line(line) for line in lines]
df = pd.DataFrame([d for d in data if d is not None])

# Convert UNIX timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Set the range for timestamp
start_timestamp = pd.to_datetime(1341100972, unit='s')
end_timestamp = pd.to_datetime(1341705593, unit='s')
df = df[(df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)]

# Bin the timestamps (e.g., hourly bins)
df['time_bin'] = df['timestamp'].dt.floor('H')

# Count the interactions in each bin
interaction_counts = df.groupby(['time_bin', 'interaction']).size().unstack().fillna(0)

# Plotting
interaction_counts.plot(kind='line', figsize=(12, 6))
plt.title('Trending of User Interactions Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Interactions')
plt.show()

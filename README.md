# Outbreak Detection

Outbreak detection by using CELF Algorithm

## Datasets

Put [higgs dataset files](https://snap.stanford.edu/data/higgs-twitter.html) in `dataset/`

## Code info

- `display_outbreak.py` Load activity information from `higgs-activity_time.txt`, display the trend of information explosion by month.
![](/Users/h0rs3/Work/Learning/LeidenUniversity/SNA/course_project/outbreak_detection/outbreak_higgs.png)
- `outbreak.py` Simulate outbreaks, as well as extract nodes from the known outbreak from higgs.

- `marginal_gain.py` penalty reduction calculation. NOT FINISHED.

- `algorithm.py` From https://github.com/hautahi/IM_GreedyCELF. NOT FINISHED.

## TODOs

- [ ] Obtain all nodes(user's ID) from four different files, as the node sets.

- [ ] How to consider cost? simulate the cost of monitoring each node.

- [ ] Define a function to calculate penalty reduction - detection time, detection likelihood.

- [ ] Greedy/CELF algorithm
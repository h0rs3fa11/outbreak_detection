# Outbreak Detection

Outbreak detection by using CELF Algorithm

## Datasets

Put [higgs dataset files](https://snap.stanford.edu/data/higgs-twitter.html) in `dataset/`

## Code info

- `display_outbreak.py` Load activity information from `higgs-activity_time.txt`, display the trend of information explosion by month.

![error](outbreak_higgs.png)

- `outbreak.py` Simulate outbreaks, as well as extract nodes from the known outbreak from higgs.

- `marginal_gain.py` penalty reduction calculation. NOT FINISHED.

- `algorithm.py` From https://github.com/hautahi/IM_GreedyCELF. NOT FINISHED.

## TODOs

- [x] (pre-processing)Combine four different graph file.

- [ ] How to consider cost? simulate the cost of monitoring each node.

- [ ] When inspecting outbreaks on Twitter, should we also consider the follower network? When an outbreak occurs through user A, are all user A's followers considered affected by the outbreak?

- [ ] Define a function to calculate penalty reduction - detection time, detection likelihood.

- [ ] Greedy/CELF algorithm.
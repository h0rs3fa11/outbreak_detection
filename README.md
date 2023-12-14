# Outbreak Detection

Detecting outbreak and optimizing objective functions by using different Algorithms:

- Naive Greedy algorithm
- CELF
- Heuristic approaches(based on inlinks, outlinks, followers, ...)

## Datasets

Put [higgs dataset files](https://snap.stanford.edu/data/higgs-twitter.html) in `dataset/`

```
higgs-mention_network.edgelist
higgs-reply_network.edgelist
higgs-retweet_network.edgelist
higgs-social_network.edgelist
```

## Usage

```
python3 main.py --help
usage: main.py [-h] [-test {1,0}] -g {mt,rt} -b BUDGET -obj {DL,DT,PA} {algorithm,heuristic} ...

Find the best placement for outbreak detection

positional arguments:
  {algorithm,heuristic}
                        sub command help
    algorithm           algorithm: choose algorithms and objective functions
    heuristic           heuristic: use heuristic methods to find placement

options:
  -h, --help            show this help message and exit
  -test {1,0}, --test {1,0}
                        Whether to use a smaller dataset for testing
  -g {mt,rt}, --graph {mt,rt}
                        indicate which graph to use, mt(mention) or rt(retweet)
  -b BUDGET, --budget BUDGET
                        total budget
  -obj {DL,DT,PA}, --objective_function {DL,DT,PA}
                        objective functions
```

## References

<a id=1>[1]</a>
M. De Domenico, A. Lima, P. Mougel and M. Musolesi. The Anatomy of a Scientific Rumor. (Nature Open Access) Scientific Reports 3, 2980 (2013).

<a id=2>[2]</a>
M. De Domenico, A. Lima, P. Mougel and M. Musolesi. The Anatomy of a Scientific Rumor. (Nature Open Access) Scientific Reports 3, 2980 (2013).

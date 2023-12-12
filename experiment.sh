of='DL'
budget=3000
graph='mt'
# test for unit cost greedy algorithm
python3 main.py -g $graph -b $budget -obj $of algorithm -algo uc-greedy
# test for variable cost greedy algorithm
python3 main.py -g $graph -b $budget -obj $of algorithm -algo cb-greedy
# test for celf
python3 main.py -g $graph -b $budget -obj $of algorithm -algo celf


# test for heuristic methods
python3 main.py -g $graph -b $budget -obj $of heuristic -f random
python3 main.py -g $graph -b $budget -obj $of heuristic -f random

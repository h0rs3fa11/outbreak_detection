# of='DL'
of=$0
# budget=100000
budget=$1
# graph='mt'
graph=$2
# test for unit cost greedy algorithm
echo "Testing for unit cost greedy algorithm"
python3 main.py -g $graph -b $budget -obj $of algorithm -algo uc-greedy
# test for variable cost greedy algorithm
echo "Testing for variable cost greedy algorithm"
python3 main.py -g $graph -b $budget -obj $of algorithm -algo cb-greedy
# test for celf
echo "Testing for celf algorithm"
python3 main.py -g $graph -b $budget -obj $of algorithm -algo celf


# test for heuristic methods
echo "Testing for heuristic approaches"
python3 main.py -g $graph -b $budget -obj $of heuristic -f random
python3 main.py -g $graph -b $budget -obj $of heuristic -f inlinks
python3 main.py -g $graph -b $budget -obj $of heuristic -f outlinks
python3 main.py -g $graph -b $budget -obj $of heuristic -f followers

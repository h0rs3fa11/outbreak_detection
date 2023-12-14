def intersect_all_sets(sets):
    if not sets:
        return set()  

    result = sets[0]
    for s in sets[1:]:
        result = result.intersection(s)

    return result

def output(algo, placement, reward, runtime): 
    return {
            'algo': algo,
            'placement': placement,
            'reward': reward,
            'runtime': runtime
    }
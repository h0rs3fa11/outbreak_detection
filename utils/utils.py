def intersect_all_sets(sets):
    if not sets:
        return set()  

    result = sets[0]
    for s in sets[1:]:
        result = result.intersection(s)

    return result

def find_minimum_activity_time(cascade):
    min_time = float('inf')  
    for activity_time in cascade.values():
        time = float(activity_time)  
        if time < min_time:
            min_time = time
    return min_time if min_time != float('inf') else None

def output(algo, placement, reward, runtime): 
    return {
            'algo': algo,
            'placement': placement,
            'reward': reward,
            'runtime': runtime
    }
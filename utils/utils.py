def intersect_all_sets(sets):
    if not sets:
        return set()  

    result = set(sets[0])
    for s in sets[1:]:
        result = result.intersection(s)

    return result

def normalize_numbers(numbers, max_value, min_value=0):
    """
    Normalizes a list of numbers to the range [0, 1].
    """
    if not numbers:
        return []

    # Avoid division by zero in case all numbers are the same
    if min_value == max_value:
        return [0.0] * len(numbers)

    return [(x - min_value) / (max_value - min_value) for x in numbers]

def output(algo, placement, reward, runtime): 
    return {
            'algo': algo,
            'placement': placement,
            'reward': reward,
            'runtime': runtime
    }
from tqdm import tqdm

def series_calc(upper_bound:float)->float:
    total = 0
    for i in tqdm(range(upper_bound+1)):
        current_prob = i/(2**i)
        total += current_prob
    
    return total

print(series_calc(1000000))
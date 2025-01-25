from typing import List

IN1 = [1,1,1,1,0,0,0,0]
IN2 = [1,1,0,0,1,1,0,0]
IN3 = [1,0,1,0,1,0,1,0]
T = [0,1,1,0,1,0,0,0]

w0 = -0.1
w1 = 0.02
w2 = 0.03
w3 = 0.04

BIAS = -1
ETA = 0.25
RUN_LIMIT = 10000

def update(weight:float, y:float, t:float, x:float)->float:
    updated_weight = weight - (ETA * (y - t) * x)
    return updated_weight

def neuron(index:int, w0:float, w1:float, w2:float, w3:float)->int:
    t = (w0 * BIAS) + (w1 * IN1[index]) + (w2 * IN2[index]) + (w3 * IN3[index])
    print(f"t: {t}")
    if t <= 0:
        return 0
    return 1

def run_test(w0, w1, w2, w3)->List[int]:
    t_out = []
    for i in range(len(T)):
        t = neuron(i, w0, w1, w2, w3)
        # print(f"Index: {i}, t = {t}")
        t_out.append(t)
    return t_out


runs = 1
solved = False
while not solved:
    test = []
    test = run_test(w0, w1, w2, w3)
    print(f"Run {runs}:\nw0: {w0}, w1: {w1}, w2: {w2}, w3: {w3}")
    print(f"Target: {T}\nTest: {test}")
    if test == T or runs > RUN_LIMIT:
        solved = True
    else:
        for i in range(len(T)):
            if T[i] != test[i]:
                w0 = update(w0, y=test[i], t=T[i], x=BIAS)
                w1 = update(w1, y=test[i], t=T[i], x=IN1[i])
                w2 = update(w2, y=test[i], t=T[i], x=IN2[i])
                w3 = update(w3, y=test[i], t=T[i], x=IN3[i])
    runs += 1

if runs < RUN_LIMIT:
    print("Solved")    
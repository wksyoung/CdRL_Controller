import numpy as np
N = 8

def uptree(weights):
    s = weights / np.sum(weights)
    seed = np.random.random()
    head = 0
    winner = None
    for i in range(len(s)):
        upper = np.sum(s[:(i+1)])
        if seed >= head and seed <= upper:
            winner = i
            break
        else:
            head = upper
    if winner is None:
        winner = np.argmax(weights)
    return winner

def Lyapunov_L1(x):
    return x.dot(x)*3

def utility(x, u):
    return x.dot(x) * 3 + u.dot(u)*0.3

def sleeping_experts(x, u, weights, agent):
    intensity = np.sum(weights)
    next_x = agent.identifier_predict(x, u)
    if Lyapunov_L1(next_x) - Lyapunov_L1(x) > 0:
        pmis = 0
        sv = np.zeros([N])
        for i in range(N):
            next_x = agent.identifier_predict(x, agent.get_actor_output(i, x))
            p = weights[i] / intensity
            if Lyapunov_L1(next_x) - Lyapunov_L1(x) > 0:
                pmis = pmis + p
                sv[i] = 1
            else:
                sv[i] = 0
        for i in range(N):
            ri = pmis / (1 + 0.01) - sv[i]
            weights[i] = weights[i] * (1+0.01)**ri
    return weights



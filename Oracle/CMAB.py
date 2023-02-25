def CMAB_max(currentP, seed_size):
    S = []

    sort_P = sorted(currentP.items(), key=lambda currentP: currentP[1], reverse=True)

    for i, (node, node_P) in enumerate(sort_P):
        if i < seed_size:
            S.append(node)

    return S

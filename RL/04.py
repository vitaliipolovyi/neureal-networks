# Initialize Markov Decision Process model
actions = (0, 1)  # actions (0=left, 1=right)
states = (0, 1, 2, 3, 4)  # states (tiles)
rewards = [-1, -1, 10, -1, -1]  # Direct rewards per state
gamma = 0.9  # discount factor
delta = 10  # Error tolerance
# Transition probabilities per state-action pair
probs = [
    [[0.9, 0.1], [0.1, 0.9], [0, 0], [0, 0], [0, 0]],
    [[0.9, 0.1], [0, 0], [0.1, 0.9], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],  # Terminating state (all probs 0)
    [[0, 0], [0, 0], [0.9, 0.1], [0, 0], [0.1, 0.9]],
    [[0, 0], [0, 0], [0, 0], [0.9, 0.1], [0.1, 0.9]],
]

it = 1

for max_iter in range(1, 10):
    # Set policy iteration parameters
    max_policy_iter = max_iter # Maximum number of policy iterations
    max_value_iter = max_iter # Maximum number of value iterations
    pi = [0 for s in states]
    V = [0 for s in states]

    tc = 0
    
    for i in range(max_policy_iter):
        # Initial assumption: policy is stable
        optimal_policy_found = True

        # Policy evaluation
        # Compute value for each state under current policy
        for j in range(max_value_iter):
            max_diff = 0  # Initialize max difference
            V_new = [0, 0, 0, 0, 0]  # Initialize values
            for s in states:

                # Compute state value
                val = rewards[s]  # Get direct reward
                for s_next in states:
                    val += probs[s][s_next][pi[s]] * (
                            gamma * V[s_next]
                    )  # Add discounted downstream values

                # Update maximum difference
                max_diff = max(max_diff, abs(val - V[s]))

                V[s] = val  # Update value with highest value
            # If diff smaller than threshold delta for all states, algorithm terminates
            if max_diff < delta:
                break
        
        # Policy iteration
        # With updated state values, improve policy if needed
        for s in states:

            val_max = V[s]
            for a in actions:
                val = rewards[s]  # Get direct reward
                for s_next in states:
                    val += probs[s][s_next][a] * (
                        gamma * V[s_next]
                    )  # Add discounted downstream values

                # Update policy if (i) action improves value and (ii) action different from current policy
                if val > val_max and pi[s] != a:
                    pi[s] = a
                    val_max = val
                    optimal_policy_found = False

        # If policy did not change, algorithm terminates
        if optimal_policy_found:
            break
        
        tc = tc + 1
        
    print('it = %d' % it)
    print('tc = %d' % tc)
    print(V)
    print(pi)
    print('---------------')
    
    it = it + 1
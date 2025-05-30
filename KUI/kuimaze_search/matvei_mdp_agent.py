"""
env = kuimaze.MDPMaze(MAP1)
env2 = kuimaze.MDPMaze(MAP1, probs=(0.8, 0.1, 0.1, 0.0))
env.get_all_states()
env.is_terminal_state(State(0, 0))
actions = tuple(env.get_actions(State(1,0)))
env2.get_next_states_and_probs(State(1,0), actions[0])

policy = dict( state: action )
"""

import kuimaze
from kuimaze import State
from kuimaze import MDPMaze

import copy

# нужно ли восстанавливать actions ?


# value iterration
def find_policy_via_value_iteration(problem, discount_factor, epsilon):
    print("sDPJKG""dsjGgfds;lFKMknhgjopesfKLD:GmndkljseofkpaDGLj;okpf[d:G")
    states = problem.get_all_states()  # gets all states
    policy = {state: 0 for state in states}  # policy init

    # main loop
    while True:
        oldPolicy = policy.copy()  # values V
        delta_err = 0  # delta error - max difference

        for state in states:
            if problem.is_goal_state(state):  # skips final states
                policy[state] = None
                continue

            actions = tuple(problem.get_actions(state))  # gets all actions
            max_value = float("-inf")

            # calculating the max value
            for action in actions:
                pos_states_and_probs = problem.get_next_states_and_probs(state, action)  # gets possible states and actions from the state
                num_of_pos_states_and_probs = len(pos_states_and_probs)

                # separation into new states and probabilities
                pos_states = [
                    pos_states_and_probs[i][0]
                    for i in range(num_of_pos_states_and_probs)
                ]
                pos_probs = [
                    pos_states_and_probs[i][1]
                    for i in range(num_of_pos_states_and_probs)
                ]

                # calculation of the current value
                value = sum(
                    [
                        pos_probs[i] * oldPolicy[pos_states[i]]
                        for i in range(num_of_pos_states_and_probs)
                    ]
                )

                # resetting the max value
                if value > max_value:
                    max_value = value

            # setting max value of state
            policy[state] = problem.get_reward(state) + discount_factor * max_value
            # print("P",policy)
            # print("sDPJKG"dsjGgfds;lFKMknhgjopesfKLD:GmndkljseofkpaDGLj;okpf[d:G"])

            # resetting max difference
            if abs(policy[state] - oldPolicy[state]) > delta_err:
                delta_err = policy[state] - oldPolicy[state]

        # break point (if converged)
        if delta_err < epsilon * (1 - discount_factor) / discount_factor:
            break

    # print("value iterration", policy)
    return policy


# policy iterration
def find_policy_via_policy_iteration(problem, discount_factor):
    def policy_eval(policy, values, problem, discount_factor):
        """
        sets value for each state
        returns dict {state : value}
        """

        # iterration through states
        for state in values.keys():
            # gets possible states and actions from the state
            pos_policy_states_and_probs = problem.get_next_states_and_probs(state, policy[state])
            num_of_pos_policy_states_and_probs = len(pos_policy_states_and_probs)

            # separation into new states and probabilities
            pos_policy_states = [
                pos_policy_states_and_probs[i][0]
                for i in range(num_of_pos_policy_states_and_probs)
            ]
            pos_policy_probs = [
                pos_policy_states_and_probs[i][1]
                for i in range(num_of_pos_policy_states_and_probs)
            ]

            # setting values
            values[state] = sum(
                [
                    # P(s'|s,a) * (R(s) + disc.factor * V_old(s'))
                    pos_policy_probs[i]
                    * (
                        problem.get_reward(state)
                        + discount_factor * values[pos_policy_states[i]]
                    )
                    for i in range(num_of_pos_policy_states_and_probs)
                ]
            )
        return values

    states = problem.get_all_states()  # gets all states

    policy = {state: tuple(problem.get_actions(state))[0] for state in states}  # sets start policy
    values = {state: 0 for state in states}  # sets start values

    unchanged = True  # change trigger

    # main loop
    while unchanged:
        values = policy_eval(policy, values, problem, discount_factor)  # sets value for each state
        unchanged = True

        # iterrates through states
        for state in states:
            actions = tuple(problem.get_actions(state))  # gets all actions

            # gets possible states and actions from the state
            pos_policy_states_and_probs = problem.get_next_states_and_probs(state, policy[state])
            num_of_pos_policy_states_and_probs = len(pos_policy_states_and_probs)

            # separation into new states and probabilities
            pos_policy_states = [
                pos_policy_states_and_probs[i][0]
                for i in range(num_of_pos_policy_states_and_probs)
            ]
            pos_policy_probs = [
                pos_policy_states_and_probs[i][1]
                for i in range(num_of_pos_policy_states_and_probs)
            ]

            # calculating policy sum
            sum_policy_states = sum(
                [
                    pos_policy_probs[i] * values[pos_policy_states[i]]
                    for i in range(num_of_pos_policy_states_and_probs)
                ]
            )

            # calculating max sum through for new action
            max_sum_new_states = float("-inf")
            max_action = actions[0]

            for action in actions:
                # gets possible states and actions from the state
                pos_new_states_and_probs = problem.get_next_states_and_probs(state, action)
                num_of_pos_new_states_and_probs = len(pos_new_states_and_probs)

                # separation into new states and probabilities
                pos_new_states = [
                    pos_new_states_and_probs[i][0]
                    for i in range(num_of_pos_new_states_and_probs)
                ]
                pos_new_probs = [
                    pos_new_states_and_probs[i][1]
                    for i in range(num_of_pos_new_states_and_probs)
                ]

                # calculating sum
                sum_new_states = sum(
                    [
                        pos_new_probs[i] * values[pos_new_states[i]]
                        for i in range(num_of_pos_new_states_and_probs)
                    ]
                )

                # resetting max sum
                if sum_new_states > max_sum_new_states:
                    max_sum_new_states = sum_new_states
                    # print("sum = ", sum_new_states)
                    max_action = action

            # resetting policy actions
            if max_sum_new_states > sum_policy_states:
                policy[state] = max_action
                # print("change")
                unchanged = False

    # print("policy iterration", policy)
    return policy


import kuimaze
from kuimaze import State
from kuimaze import MDPMaze

import copy

def find_policy_via_value_iteration(problem, discount_factor, epsilon):

    all_states=[]
    all_states=problem.get_all_states()
    poliсy={}
    value={}
    # policy and value for each state
    for state in all_states:
        value[state]=0
    while True :
        min_difference=0
        # minimal separation between old value and value in next iteraction 
        old_value=value.copy()

        for state in all_states:
            reward=problem.get_reward(state)
            if (problem.is_terminal_state(state)):
                value[state]=reward
                continue
            actions=tuple(problem.get_actions(state))

            max_value = float("-inf")
            best_policy=''
            # these values will be saved in value{} and policy{}
            for action in actions:
                next_states_and_probs=problem.get_next_states_and_probs(state, action)
                action_value=0
                for i in range(len(next_states_and_probs)):
                    action_value+=value[next_states_and_probs[i][0]]*next_states_and_probs[i][1]
                if max_value < action_value:
                    max_value=action_value
                    best_policy=action
            poliсy[state]=best_policy
            value[state]=reward+discount_factor*max_value
            min_difference=max(min_difference, abs(value[state]-old_value[state]))

        if min_difference<epsilon:
            return poliсy    
        



def find_policy_via_policy_iteration(problem, discount_factor):
    all_states=[]
    all_states=problem.get_all_states()
    policy={}
    value={}
    for state in all_states:
        value[state]=0
        policy[state]=tuple(problem.get_actions(state))[0]
    while True:
        for state in all_states:
            reward=problem.get_reward(state)
            if (problem.is_terminal_state(state)):
                value[state]=reward
                continue

            next_states_and_probs=problem.get_next_states_and_probs(state, policy[state])
            action_value=0
            for i in range(len(next_states_and_probs)):
                action_value+=value[next_states_and_probs[i][0]]*next_states_and_probs[i][1]
            value[state]=reward+discount_factor*action_value

        old_policy=policy.copy()
        for state in all_states:
            if (problem.is_terminal_state(state)):
                policy[state]=None
                continue

            reward=problem.get_reward(state)
            actions=tuple(problem.get_actions(state))
            max_value = float("-inf")
            best_policy=''
            for action in actions:
                next_states_and_probs=problem.get_next_states_and_probs(state, action)
                action_value=0
                for i in range(len(next_states_and_probs)):
                    action_value+=value[next_states_and_probs[i][0]]*next_states_and_probs[i][1]
                if max_value < action_value:
                    max_value=action_value
                    best_policy=action
            policy[state]=best_policy
        control=1
        for state in all_states:
            if policy[state]!=old_policy[state]:
                control=0
        if control==1:
            print(value)
            return policy
    


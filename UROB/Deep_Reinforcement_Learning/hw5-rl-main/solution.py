import torch

def policy_gradient_loss_simple(logp: torch.Tensor, tensor_r: torch.Tensor) -> torch.Tensor:
    """
    Given the log-probabilities of the policy and the rewards, compute the scalar loss
    representing the policy gradient.

    Args:
        logp:    (T, N) tensor of log-probabilities of the policy
                 - T = episode length (timesteps per trajectory)
                 - N = number of trajectories (batch of parallel simulations)
        tensor_r: (T, N) tensor of rewards, detached from any computation graph

    Returns:
        policy_loss: scalar tensor representing the policy gradient loss
    """

    with torch.no_grad():
        returns_per_trajectory = tensor_r.sum(dim=0, keepdim=True) 
    weighted_logp = logp * returns_per_trajectory  
    policy_loss = - (1.0 / (logp.shape[0] * logp.shape[1])) * weighted_logp.sum()

    return policy_loss


def discount_cum_sum(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Given the rewards and the discount factor gamma, compute the discounted cumulative sum of rewards.
    The cumulative sum follows the reward-to-go formulation. This means we want to compute the discounted
    trajectory returns at each timestep. We do that by calculating an exponentially weighted
    sum of (only) the following rewards.
    i.e.
    $R(\tau_i, t) = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}$

    Args:
        rewards: (T, N) tensor of rewards
        gamma: discount factor

    Returns:
        discounted_cumulative_returns: (T, N) tensor of discounted cumulative returns
    """
    # TODO: implement the discounted cummulative sum, i.e. the discounted returns computed from rewards and gamma

    T, N = rewards.shape 
    discounted_cum_returns = torch.zeros_like(rewards)
    discounted_cum_returns[-1] = rewards[-1]
    for t in reversed(range(T - 1)):
        discounted_cum_returns[t] = rewards[t] + gamma * discounted_cum_returns[t + 1]
    return discounted_cum_returns




def policy_gradient_loss_discounted(logp: torch.Tensor, tensor_r: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Given the policy log0probabilities, rewards and the discount factor gamma, compute the
    policy gradient loss using discounted returns.

    Args:
        logp: (T, N) tensor of log-probabilities of the policy
        tensor_r: (T, N) tensor of rewards, detached from any computation graph
        gamma: discount factor

    Returns:
        policy_loss: scalar tensor representing the policy gradient loss
    """
    # with torch.no_grad():
    # TODO: compute discounted returns of the trajectories from the reward tensor, then compute the policy gradient
    with torch.no_grad():
        discounted_returns=discount_cum_sum(tensor_r,gamma)


    weighted_logp = logp * discounted_returns
    policy_loss = -weighted_logp.mean()
    return policy_loss
    



def policy_gradient_loss_advantages(logp: torch.Tensor, advantage_estimates: torch.Tensor) -> torch.Tensor:
    """
    Given the policy log-probabilities and the advantage estimates, compute the policy gradient loss

    Args:
        logp: (T, N) tensor of log-probabilities of the policy
        advantage_estimates: (T, N) tensor of advantage estimates

    Returns:
        policy_loss: scalar tensor representing the policy gradient loss
    """
    # TODO: compute the policy gradient estimate using the advantage estimate weighting

    policy_gradient_loss_advantages = (logp * advantage_estimates)
    value_loss_mean = -policy_gradient_loss_advantages.mean()
    return value_loss_mean




def value_loss(values: torch.Tensor, value_targets: torch.Tensor) -> torch.Tensor:
    """ 
    Given the values and the value targets, compute the value function regression loss
    """
    # TODO: compute the value function L2 loss
    value_loss = (values - value_targets)**2
    value_loss_mean = value_loss.mean()
    return value_loss_mean




def ppo_loss(p_ratios, advantage_estimates, epsilon):
    """ 
    Given the probability ratios, advantage estimates and the clipping parameter epsilon, compute the PPO loss
    based on the clipped surrogate objective
    """
    # TODO: compute the PPO loss

    clipped_ratios = torch.clamp(p_ratios, 1.0 - epsilon, 1.0 + epsilon)
    obj_unclipped = advantage_estimates * p_ratios
    obj_clipped   = advantage_estimates * clipped_ratios
    ppo_obj = torch.min(obj_unclipped, obj_clipped)
    ppo_loss = -ppo_obj.mean()
    return ppo_loss




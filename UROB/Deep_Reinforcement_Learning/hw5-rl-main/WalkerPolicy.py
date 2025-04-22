import torch
from torch import nn
from torch.distributions import Normal


class WalkerPolicy(nn.Module):
    def __init__(self, state_dim: int = 29, action_dim: int = 8, load_weights: bool = False):
        super().__init__()

        # Define the policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Learnable log_std parameter for the action distribution
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)  # Initialize log_std

        if load_weights:
            self.load_weights()

    def forward(self, states: torch.Tensor) -> tuple:
        """
        Forward pass through the policy network.

        Args:
            states: (N, state_dim) tensor

        Returns:
            dist: torch.distributions.Normal distribution
            mu: (N, action_dim) mean actions
            sigma: (N, action_dim) standard deviation of actions
        """
        mu = self.policy_net(states)  
        log_std = self.log_std.expand_as(mu) 
        sigma = torch.exp(log_std) 
        dist = Normal(mu, sigma) 
        return dist, mu, sigma

    def determine_actions(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get deterministic actions (mean of the distribution).

        Args:
            states: (N, state_dim) tensor

        Returns:
            actions: (N, action_dim) tensor
        """
        with torch.no_grad():
            _, mu, _ = self.forward(states)
            return mu  # Return the mean action

    def sample_actions_and_log_prob(self, states: torch.Tensor) -> tuple:
        """
        Sample actions and return log probabilities.

        Args:
            states: (N, state_dim) tensor

        Returns:
            actions: (N, action_dim) tensor
            log_probs: (N, 1) tensor
        """
        dist, _, _ = self.forward(states)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        return actions, log_probs

    def save_weights(self, path: str = 'walker_weights.pt') -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str = 'walker_weights.pt') -> None:
        self.load_state_dict(torch.load(path))

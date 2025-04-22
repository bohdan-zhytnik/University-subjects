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

        # Define a learnable log_std parameter
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)  # Initialize log_std

        if load_weights:
            self.load_weights()

    def forward(self, states: torch.Tensor):
        """
        Forward pass through the policy network.
        
        This returns:
          1) A Normal distribution object for sampling actions or computing log-probs.
          2) The action means (mu) for each state.
          3) The action std-dev (sigma) for each state.

        Args:
            states: (N, state_dim) tensor of input states

        Returns:
            dist:   torch.distributions.Normal of shape (N, action_dim).
            mu:     (N, action_dim) - the computed action means.
            sigma:  (N, action_dim) - the computed (positive) standard deviations.
        """
        # Compute action means
        mu = self.policy_net(states)  # shape (N, action_dim)

        # Expand or broadcast the log-std to match (N, action_dim)
        log_std = self.log_std.expand_as(mu)  # shape (N, action_dim)

        # Convert log-std to std
        sigma = torch.exp(log_std)  # shape (N, action_dim)

        # Create a Normal distribution for each action dimension
        dist = Normal(mu, sigma)  # Batched distribution: shape (N, action_dim)

        return dist, mu, sigma

    def determine_actions(self, states: torch.Tensor) -> torch.Tensor:
        """
        By default, we might return the *deterministic* action (mu).
        For training, you might want to sample: dist.sample().
        
        Args:
            states: (N, state_dim) tensor

        Returns:
            (N, action_dim) actions tensor.
        """
        with torch.no_grad():
            dist, mu, _ = self.forward(states)
            # Deterministic action (no exploration):
            actions = mu
            return actions

    def sample_actions_and_log_prob(self, states: torch.Tensor) -> tuple:
        """
        Sample actions and return log-probabilities.

        Args:
            states (torch.Tensor): (N, state_dim) tensor

        Returns:
            actions (torch.Tensor): (N, action_dim) tensor
            log_probs (torch.Tensor): (N, 1) tensor
        """
        dist, _, _ = self.forward(states)  # Get distribution from forward pass
        actions = dist.sample()  # Sample actions
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)  # Compute log-prob
        return actions, log_probs

    def save_weights(self, path: str = 'walker_weights.pt') -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str = 'walker_weights.pt') -> None:
        self.load_state_dict(torch.load(path))

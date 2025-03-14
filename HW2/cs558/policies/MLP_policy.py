import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs558.infrastructure import pytorch_util as ptu
from cs558.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, 
                size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        # Note that the default policy above defines parameters for both mean and variance.
        # It is up to you whether you want to use both to sample actions (recommended) or just the mean.
        observation = ptu.from_numpy(observation)

        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            action = action_distribution.sample()
        else:
            mean = self.mean_net(observation)
            std = torch.exp(self.logstd)
            action_distribution = distributions.Normal(mean, std)
            action = action_distribution.sample()
        
        return ptu.to_numpy(action)

        # raise NotImplementedError

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)

        if self.discrete:
            logits = self.logits_na(observations)
            loss = F.cross_entropy(logits, actions.long())
        else:
            pred_actions = self.mean_net(observations)
            loss = self.loss(pred_actions, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        if self.discrete:
            logits = self.logits_na(observation)
            return logits
        else:
            mean = self.mean_net(observation)
            std = torch.exp(self.logstd)
            return mean, std


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss
        # Note that you do not have to use MSELoss (defined in init), and in fact it may be preferable to use something else if you are trying to learn both the mean and variance.
        # loss = TODO

        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)

        if self.discrete:
            logits = self.forward(observations)
            loss = F.cross_entropy(logits, actions.long())
        else:
            mean, std = self.forward(observations)
            # loss = self.loss(mean, actions)

            # using negative log-likelihood for loss since mse was
            # giving me negative reward return averages for eval
            action_distribution = distributions.Normal(mean, std)
            loss = -action_distribution.log_prob(actions).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
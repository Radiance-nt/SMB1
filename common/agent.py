from copy import deepcopy
from typing import Optional, Any, Union, Dict, List
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import PPOPolicy, BasePolicy
from tianshou.utils.net.common import ActorCritic, MLP
import torch.nn.functional as F
# import torchvision.transforms
from common.neural import MarioNet, LatentTransition
from tianshou.utils.net.discrete import Actor, Critic


class SPRAgent(PPOPolicy):
    K = 5
    it = 0
    spr_lambda = 0.6
    spr_update_freq = 1
    spr_tau = 0.9

    def __init__(self, observation_space, action_space, hidden_size=64, args=None):
        super(BasePolicy, self).__init__()
        model = MarioNet(observation_space, hidden_size, args.device).to(args.device)
        actor = Actor(model, action_shape=action_space, hidden_sizes=[hidden_size, 64],
                      device=args.device).to(args.device)
        critic = Critic(model, [hidden_size, 64], device=args.device).to(args.device)
        actor_critic = ActorCritic(actor, critic)
        # orthogonal initialization
        try:
            for m in actor_critic.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
        except:
            pass
        optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
        dist = torch.distributions.Categorical
        super().__init__(actor,
                         critic,
                         optim,
                         dist, )
        self.model = model
        self.target_model = deepcopy(self.model)
        self.spr = args.spr
        if self.spr:
            self.spr_batch_size = args.spr_batch_size
            self.spr_lr = args.spr_lr
            self.transition = LatentTransition(hidden_size, action_space,
                                               hidden_size, hidden_size, args.device).to(args.device)
            self.online_projection = MLP(hidden_size, hidden_size, [hidden_size]).to(args.device)
            self.target_projection = deepcopy(self.online_projection)
            self.prediction = MLP(hidden_size, hidden_size, [hidden_size]).to(args.device)
            joint_model = nn.Sequential(self.transition, self.prediction, self.online_projection)
            self.spr_optimizer = torch.optim.Adam(joint_model.parameters(), lr=self.spr_lr)
            self.spr_augment = transforms.Compose(
                [transforms.RandomResizedCrop(84, scale=(0.6, 1), ratio=(0.8, 1.2)),
                 transforms.RandomAffine(degrees=5)
                 ])

    def augment(self, batch):
        if self.spr:
            obs = torch.from_numpy(batch.obs)
            batch.obs = self.spr_augment(obs).numpy()
        return batch

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any,
    ) -> Batch:
        with torch.no_grad():
            features = self.augment(batch)
        return super(SPRAgent, self).forward(features, state, **kwargs)

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        if buffer is None:
            return {}
        self.updating = True
        if self.spr:
            batch, indices = buffer.sample(self.spr_batch_size)
            spr_result = self.spr_update(batch, buffer, indices)
        else:
            spr_result = {}
        batch, indices = buffer.sample(sample_size)
        batch = self.process_fn(batch, buffer, indices)
        result = self.learn(batch, **kwargs)
        result.update(spr_result)
        self.post_process_fn(batch, buffer, indices)
        self.updating = False
        return result

    def spr_update(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Dict[str, List[float]]:
        zk, _ = self.model(self.augment(batch).obs)
        spr_loss = 0
        for k in range(1, self.K):
            batch = buffer[indices + k]
            batch = self.augment(batch)
            zk = self.transition(zk, batch.act)
            zk_pred = self.prediction(self.online_projection(zk))
            with torch.no_grad():
                sk_features, _ = self.target_model(batch.obs)
                sk_features = self.target_projection(sk_features)
            spr_loss = spr_loss - F.cosine_similarity(zk_pred, sk_features).mean()
        spr_loss = self.spr_lambda * spr_loss
        self.spr_optimizer.zero_grad()
        spr_loss.backward()
        self.spr_optimizer.step()
        self.it = (self.it + 1) % self.spr_update_freq
        if self.it % self.spr_update_freq == 0:
            self.soft_update()

        return {
            "loss/spr": spr_loss.item(),
        }

    def soft_update(self):
        for o, n in zip(self.target_model.parameters(), self.model.parameters()):
            o.data.copy_(o.data * (1.0 - self.spr_tau) + n.data * self.spr_tau)
        for o, n in zip(self.target_projection.parameters(), self.online_projection.parameters()):
            o.data.copy_(o.data * (1.0 - self.spr_tau) + n.data * self.spr_tau)

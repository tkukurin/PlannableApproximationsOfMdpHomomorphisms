
# Plannable Approximations to MDP Homomorphisms: Equivariance under Actions

PyTorch implementation of the key task from [Plannable Approximations to MDP
Homomorphisms: Equivariance under Actions](https://arxiv.org/abs/2002.11963):

> This work exploits action equivariance for representation learning in
> reinforcement learning. Equivariance under actions states that transitions in
> the input space are mirrored by equivalent transitions in latent space, while
> the map and transition functions should also commute. We introduce a
> contrastive loss function that enforces action equivariance on the learned
> representations. We prove that when our loss is zero, we have a homomorphism
> of a deterministic Markov Decision Process (MDP). Learning equivariant maps
> leads to structured latent spaces, allowing us to build a model on which we
> plan through value iteration. We show experimentally that for deterministic
> MDPs, the optimal policy in the abstract MDP can be successfully lifted to the
> original MDP. Moreover, the approach easily adapts to changes in the goal
> states. Empirically, we show that in such MDPs, we obtain better
> representations in fewer epochs compared to representation learning approaches
> using reconstructions, while generalizing better to new goals than model-free
> approaches.


## Running

Perform a training run on CPU:
```
python config.gin train --device cpu
python config.gin plan --device cpu
```


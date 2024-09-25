## If we have rollout_length = 1 and can use StateSpaces we highly optimize our learning.

#### Create the Dataset

1. Create the env over all spaces
2. Compute `num_batches` as `total_states` / `batch_size`
3. Create the whole Dataset by querying the env `num_batches` times
4. Get the reward and done signals for every possible transition
5. Compute the hetero-graph encoding for every state
   Each data point is

- Encoding of state
- Encoding of next states
- Reward for each next state
- Terminated for each next state

Could be one HeteroData object.
But RL losses expect TensorDicts
Similar process for evaluation.

#### Adapting the Agent

- Agent will have to use encoded-states instead of state
    - (encoded state, encoded successor state) instead of transition
- Output index instead of actual state
- Implement LightningModule

#### Adapting the Losses

- Can only use $v(s), v(s'), R_{t+1}, done, \pi(s'\mid s)$ signals

#### Adapting the rest

- Use Lightning.Trainer
- Adapt Evaluation Hooks
- Reverting back to lightning logging

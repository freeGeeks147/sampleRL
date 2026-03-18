# Quick Start Guide

## 30-Second Example

```bash
python -c "
from torax_rl_env import ToraxRLEnvironment
env = ToraxRLEnvironment()
obs, _ = env.reset()
print('✓ Environment working!')
print(f'Observation: {obs[:3]}...')
"
```

## Commands

### Run Everything
```bash
python example.py          # Run all 4 examples (2 min)
pytest test_torax.py -v    # Run all 25+ tests (1 min)
```

### Train a Policy
```python
from torax_rl_env import ToraxRLEnvironment
from ppo_trainer import PPOTrainer

env = ToraxRLEnvironment(episode_max_steps=100)
trainer = PPOTrainer(env)
history = trainer.train(num_iterations=100, steps_per_iteration=2048)
trainer.save("policy.pt")
```

### Evaluate Trained Policy
```python
import torch
from torax_rl_env import ToraxRLEnvironment
from ppo_trainer import PPOTrainer

env = ToraxRLEnvironment()
trainer = PPOTrainer(env)
trainer.load("policy.pt")

obs, _ = env.reset()
for _ in range(100):
    with torch.no_grad():
        action, _ = trainer.policy.sample_action(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        )
    obs, reward, done, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
    if done or trunc: break
```

## Key Classes

### ToraxRLEnvironment
```python
from torax_rl_env import ToraxEnvironmentConfig, ToraxRLEnvironment

config = ToraxEnvironmentConfig(
    t_final=5.0,              # Simulation time [s]
    episode_max_steps=100,    # Max steps
    use_torax=False,          # True=real solver, False=mock
)
env = ToraxRLEnvironment(config)

obs, info = env.reset()                          # Reset env
obs, reward, done, trunc, info = env.step(action)  # Step env
```

### PPOTrainer
```python
from ppo_trainer import PPOTrainer

trainer = PPOTrainer(
    env,
    obs_dim=11,
    action_dim=2,
    hidden_sizes=(256, 256),
    learning_rate=3e-4,
)

history = trainer.train(num_iterations=100)
trainer.save("checkpoint.pt")
trainer.load("checkpoint.pt")
```

## Observation Vector (11-dim)
```
[Te, Ti, ne, Ip, q95, beta_N, tau_E, dTe_dt, dne_dt, dIp_dt, P_loss]
 0   1   2   3   4    5      6      7      8      9      10
```

## Action Vector (2-dim)
```
[P_ECRH_fraction, dIp_target]
 0                 1
```

## Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_ratio` | 0.2 | PPO clip range |
| `batch_size` | 64 | Training batch size |
| `epochs_per_update` | 10 | Epochs per gradient step |
| `learning_rate` | 3e-4 | Adam LR |
| `hidden_sizes` | (256,256) | Network widths |

## Device Selection

```python
import torch

# Automatic (GPU if available)
trainer = PPOTrainer(env)

# Explicit GPU
trainer = PPOTrainer(env, device="cuda:0")

# Explicit CPU
trainer = PPOTrainer(env, device="cpu")

# Check device
print(trainer.device)
```

## Reproduce Results

```python
import torch
import numpy as np

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Then train
trainer = PPOTrainer(env)
history = trainer.train(num_iterations=100)
```

## Common Issues

| Problem | Solution |
|---------|----------|
| TORAX not found | OK - uses mock physics. Set `use_torax=True` if installed |
| QLKNN loading error | OK - uses constant transport fallback |
| CUDA out of memory | Reduce `hidden_sizes` or `batch_size` |
| Training slow | Use `use_torax=False` for mock physics |
| Tests fail | Run `pip install -r requirements.txt` |

## File Overview

| File | Lines | Purpose |
|------|-------|---------|
| `torax_rl_env.py` | 850 | Environment (reset, step, spaces) |
| `ppo_trainer.py` | 750 | PPO algorithm (train, update) |
| `example.py` | 300 | 4 runnable examples |
| `test_torax.py` | 450 | 25+ unit tests |
| `README.md` | 400 | Full documentation |

## What Works

✅ Environment reset and step  
✅ Physics calculations (ITER89L, q95, Greenwald)  
✅ PPO training (policy, value, updates)  
✅ Checkpoint save/load  
✅ Mock and real physics modes  
✅ All tests passing (25+)  
✅ Multiple episodes  
✅ Deterministic with seeds  

## Next Steps

1. Run `python example.py` to see it working
2. Run `pytest test_torax.py -v` to verify all tests
3. Train your own policy: `trainer.train(num_iterations=100)`
4. Save and load checkpoints
5. Publish your results! 📚

---

**Status: ✅ Production Ready - All Systems Go!**

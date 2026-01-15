# Chrome Dino RL Training Plan

## Overview
Train a reinforcement learning agent to play the Chrome Dinosaur game using DQN (Deep Q-Network).

## Phase 1: Implement Game Over Detection (15 minutes)

### Task: Add frame comparison logic to `_is_game_over()`
- Store previous frame in the environment
- Compare current frame to previous frame (left portion of screen to avoid reload button)
- If frames are identical/nearly identical → game over
- Use simple pixel difference or numpy array comparison
- Check for 2-3 consecutive identical frames to avoid false positives

### Test game over detection:
- Run a few episodes manually
- Verify it correctly detects when dino crashes
- Adjust threshold if needed

## Phase 2: Improve Environment (30 minutes)

### 1. Add automatic restart
- After game over detected, press Space to restart
- Add small delay for reload button to appear

### 2. Better reward structure (optional but recommended)
- Keep survival reward at +1 per frame
- Game over penalty at -100
- Maybe add frame count to track progress

### 3. Add monitoring
- Episode counter
- Best score/longest survival
- Print stats periodically

## Phase 3: Training Setup (15 minutes)

### 1. Configure DQN parameters
```python
model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    buffer_size=50000,
    learning_starts=1000,  # Start learning after 1k steps
    target_update_interval=1000,
    tensorboard_log="./dino_tensorboard/"
)
```

### 2. Add model checkpointing
- Save model every N timesteps
- Can resume training if interrupted

### 3. Add evaluation callback
- Test performance periodically
- Track improvement over time

## Phase 4: Run Training

### Initial test run (5-10 minutes)
```bash
uv run python main.py --select-region
```
- Train for 10k-50k timesteps
- Verify everything works end-to-end
- Check that game over detection and restart work

### Full training run (hours/overnight)
- 500k-1M timesteps
- Monitor with TensorBoard: `tensorboard --logdir ./dino_tensorboard/`
- Let it run unattended

## Phase 5: Evaluation
- Load best model
- Test on fresh episodes
- Measure average survival time/score

## Implementation Priority

1. **First**: Implement `_is_game_over()` with frame comparison
2. **Second**: Add automatic restart after game over
3. **Third**: Add basic monitoring/logging
4. **Fourth**: Start training!

## Key Insights

- **Game over detection**: Screen stops changing when game ends. Compare frames from the left portion of screen (excluding center where reload button appears).
- **No need for complex detection**: Simple frame comparison is sufficient and robust.
- **Frame stacking**: Already implemented (4 frames) for temporal information.

## Expected Results

- Initial random agent: Dies within seconds
- After 10k steps: May learn to jump occasionally
- After 100k steps: Should survive longer stretches
- After 500k+ steps: Should be able to play reasonably well

## Notes

- Use `--select-region` flag to select game area before training
- Region selection now works correctly with HiDPI displays
- Frames are captured at physical resolution and scaled appropriately

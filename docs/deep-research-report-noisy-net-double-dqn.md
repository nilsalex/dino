# NoisyNet with Double DQN: Research and Implementation

**Background:** Deep Q‑learning (DQN) often suffers from limited exploration and value overestimation.  *Double DQN* (DDQN) addresses overestimation by decoupling action selection and evaluation【21†L344-L353】.  *Noisy Networks (NoisyNet)* replace ε‑greedy exploration with learned, parameterized noise in network weights【20†L297-L304】.  Fortunato et al. introduced NoisyNet-DQN by sampling new noise in the fully-connected layers each step (dropping ε‑greedy)【20†L297-L304】.  They recommend factorized Gaussian noise (initial noise scale σ₀≈0.5)【53†L376-L379】.  In practice one sets ε=0 (fully greedy) when using NoisyNet【10†L43-L50】, and uses standard DDQN updates (target = r + γ Qₜ(s′, argmaxₐ Q(s′,a;θ); θ⁻)).  Typical hyperparameters (from open implementations) are a learning rate ≈1e-4, batch size ~32, replay buffer ~100k, warm-up ~10k【57†L642-L645】.

## Research on NoisyNet + Double DQN

- **Rainbow DQN (Hessel et al. 2017):** This seminal work combined *many* DQN improvements, explicitly using both DDQN and NoisyNet in one agent.  Rainbow’s authors note “Double DQN… decoupl[es] action selection and evaluation” and “Noisy DQN uses stochastic network layers for exploration”【8†L36-L45】.  In practice they ran NoisyNet agents with ε=0 (fully greedy)【10†L43-L46】.  Rainbow achieved state-of-the-art Atari scores by combining NoisyNet and Double DQN (among other enhancements)【8†L36-L45】【10†L43-L46】.  

- **Fortunato et al. 2017 (NoisyNet):** The original NoisyNet paper describes *NoisyNet-DQN* and *NoisyNet-Dueling* by removing ε‑greedy and inserting noisy linear layers【20†L297-L304】.  They reported large improvements across Atari games (e.g. median score gains of 48% over DQN)【53†L440-L448】.  They also specify implementation details: replace each FC layer with a noisy layer and resample its noise every optimization step【20†L297-L304】, initialize factorized noise with σ₀≈0.5【53†L376-L379】, and use the same CNN architecture as vanilla DQN.  (Fortunato did **not** use Double DQN in that work, but the NoisyNet recipe is directly applicable to DDQN.)  

- **NoisyNet-DDQN (Recent Studies):** Several recent papers explicitly fuse NoisyNet with Double DQN. For example, Hu *et al.* (2025) propose **IoT-ONDDQN**, a “NoisyNet Double DQN” for network intrusion detection.  They report that replacing ε‑greedy with NoisyNets in a DDQN architecture “dynamically balances exploration/exploitation” and yields ~22% faster convergence than a standard (ε‑greedy) DQN【4†L141-L146】.  Similarly, Wu *et al.* (2026) introduce an *improved NoisyNet DQN* for UAV navigation that **combines NoisyNet and Double DQN**.  They note it “reduces Q-value overestimation by combining double DQN” and show the resulting agent “achieves faster convergence and fewer steps” than vanilla DQN or Noisy DQN alone【48†L775-L780】.  These works confirm that NoisyNet exploration can be effectively integrated with DDQN to improve learning speed and stability【4†L141-L146】【48†L775-L780】.

- **Other Applications:** Variants like NROWAN-DQN (2022) modify NoisyNet to reduce instability.  Han *et al.* show their NoisyNet variant outperforms standard NoisyNet-DQN, DDQN, and DQN in a noisy task, indicating NoisyNet can improve on DDQN but may need stabilization【31†L234-L241】.  Li *et al.* (2018) applied NoisyNet-DQN to power network reconfiguration and found it “automatically adjusts exploration” and yields better loss/voltage outcomes than ε‑greedy DDQN【40†L2254-L2261】.  (They use NoisyNet-DQN, but *mention* Double-DQN as a baseline.)  In summary, while relatively few papers focus *solely* on NoisyNet+DoubleDQN, Rainbow and recent domain-specific works show this combination is effective【8†L36-L45】【4†L141-L146】【48†L775-L780】.

## Implementations and Code Examples

Several open-source implementations make it easy to combine NoisyNets with Double DQN:

- **TensorFlow / Baselines:** Andrew Liao’s NoisyNet-DQN repo (based on OpenAI Baselines) adds noisy linear layers.  For example, their README shows running 
  ```
  python train.py --env Breakout --no-double-q --noisy --save-dir MODEL_PATH
  ```
  to train a *NoisyNet DDQN* agent【55†L342-L345】.  (Omitting `--no-double-q` would disable Double DQN, so the default is Double DQN.)  

- **PyTorch Repos:** The [smitkiri DQN variants repo](https://github.com/smitkiri/VariationsOfDQN) implements **Vanilla DQN, DDQN, Dueling DQN, and NoisyNet DQN**【13†L281-L290】.  On CartPole-v1 it reports NoisyNet-DQN converged in only ~2,000 episodes versus ~10,000 for Double DQN【13†L329-L337】.  Another example is XinJingHao’s *Noisy-Duel-DDQN-Atari-PyTorch* repo.  This code uses command-line flags to enable each feature.  For instance: 

  ```
  # Run Double DQN (no Noisy)
  python main.py --Double True  --Noisy False
  # Run Noisy DQN (no Double)
  python main.py --Double False --Noisy True
  # Run Double DQN with NoisyNet (and optional dueling)
  python main.py --Double True  --Noisy True  --Duel True
  ```
  【16†L309-L317】.  

- **Libraries:** PyTorch Lightning’s *Lightning-Bolts* provides a `NoisyDQN` model (built on Fortunato et al.) and a `DuelingDQN` model, where hyperparameters default to things like `lr=1e-4, batch_size=32, replay_size=100000, warm_start_size=10000`【57†L642-L645】.  These classes simply replace the final FC layers with `NoisyLinear` (factorized noise) and set ε=0.  Their docs note that Noisy DQN “is more stable and converges faster” than vanilla DQN【57†L789-L796】.  In practice you can copy and modify these implementations or use open AI Baselines (which now includes Double DQN by default) and add noisy layers in place of the last layers.

## Practical Tips and Hyperparameters

- **Layer Noise:** Replace each dense layer (often just the last few layers) with a `NoisyLinear` layer.  Factorized Gaussian noise is typical (fewer parameters)【53†L376-L379】.  Initialize weight noise scale σ₀≈0.5 as recommended【53†L376-L379】.  During training, sample new noise each optimization step (fix noise during a batch of transitions).

- **Exploration:** Set ε = 0 (no ε‑greedy) when using NoisyNet【10†L43-L46】.  The network’s internal noise drives exploration.  (If needed, you can anneal σ₀ over time or add a small ε, but many use purely greedy policy with noise.)

- **Double DQN Update:** Use the standard DDQN target:  
  `target = r + γ · Q_target(s′, argmaxₐ Q(s′, a; θ); θ⁻)`.  
  The only change is that Q and Q_target networks have noisy layers.  Sync the target network periodically as usual.

- **Architecture:** Use the same CNN/MLP as your base DQN.  If you use a dueling network, put noise in the advantage/value heads【20†L299-L308】.  Rainbow and implementations typically apply noise to all fully-connected layers after convolution.

- **Hyperparameters:** Aside from noise, keep DQN hyperparameters as baseline.  For example, common defaults (as in Lightning-Bolts) are: learning rate ~1e-4, discount γ≈0.99, batch size ~32, replay capacity ~100k, warm-up steps ~10k【57†L642-L645】.  NoisyNet often converges faster, so you may not need as large replay or as long training.  Be prepared for slightly higher compute per update (sampling noise), but wall‑clock time remains similar【40†L2132-L2140】【40†L2145-L2152】.

- **Empirical Results:** Across implementations, NoisyNet often yields faster learning.  For example, a NoisyNet agent in Pong reached +ve reward around 250k frames vs. 400k for DQN【30†L36-L40】.  In one open benchmark, NoisyNet-DQN needed ~2,000 episodes on CartPole-v1 to converge, whereas Double DQN needed ~10,000【13†L329-L337】.  In network control tasks, NoisyNet-DDQN converged ~22% faster than ε‑greedy DQN【4†L141-L146】 and achieved higher final reward.  These suggest that combining NoisyNet with DDQN can significantly improve exploration efficiency and stability.

In summary, **Yes – several recent works and codebases integrate NoisyNet with Double DQN**.  Rainbow DQN first demonstrated the idea, and newer papers (e.g. IoT‑ONDDQN, Wu *et al.*) explicitly build “NoisyNet DDQN” agents【4†L141-L146】【48†L775-L780】.  Many code repositories (TensorFlow, PyTorch) already include flags or classes for NoisyNets alongside double-Q.  To implement it yourself, insert noisy layers as per Fortunato et al. and train with the standard DDQN loss; as cited above, this approach often yields faster convergence and more robust learning【20†L297-L304】【8†L36-L45】.

**References and Code:** See the cited papers and repos for details. For example, Rainbow DQN【8†L36-L45】, Fortunato *et al.*’s NoisyNet【20†L297-L304】【53†L376-L379】, and Hu *et al.*’s IoT‑ONDDQN【4†L141-L146】.  For implementation, consult open-source code such as the TensorFlow NoisyNet-DQN repo【55†L342-L345】 or PyTorch examples【16†L309-L317】【13†L281-L290】. These provide ready-made modules and usage patterns for Double DQN + NoisyNet in practice.  


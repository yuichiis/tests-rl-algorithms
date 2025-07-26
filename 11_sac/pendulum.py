# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple, deque
from itertools import count
from typing import List, Tuple, Dict, Optional, Callable, Any
import copy

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal # Use Normal distribution for continuous actions

torch.set_default_tensor_type(torch.FloatTensor)  # Set default to float32

# Import Gymnasium
try:
    import gymnasium as gym
except ImportError:
    print("Gymnasium not found. Please install using 'pip install gymnasium' or 'pip install gym[classic_control]'")
    gym = None

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#%matplotlib inline

# Instantiate the Pendulum environment
if gym is not None:
    try:
        env = gym.make('Pendulum-v1')
        env.reset(seed=seed)
        env.action_space.seed(seed)

        n_observations_sac = env.observation_space.shape[0]
        n_actions_sac = env.action_space.shape[0]
        action_low_sac = env.action_space.low[0]
        action_high_sac = env.action_space.high[0]

        print(f"Pendulum Environment:")
        print(f"State Dim: {n_observations_sac}")
        print(f"Action Dim: {n_actions_sac}")
        print(f"Action Low: {action_low_sac}")
        print(f"Action High: {action_high_sac}")
    except Exception as e:
        print(f"Error creating Gymnasium environment: {e}")
        n_observations_sac = 3
        n_actions_sac = 1
        action_low_sac = -2.0
        action_high_sac = 2.0
        env = None
else:
    print("Gymnasium not available. Cannot create Pendulum environment.")
    n_observations_sac = 3
    n_actions_sac = 1
    action_low_sac = -2.0
    action_high_sac = 2.0
    env = None

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-6 # Small number for numerical stability

class ActorNetworkSAC(nn.Module):
    """ Stochastic Gaussian Actor Network for SAC """
    def __init__(self, n_observations: int, n_actions: int, action_high_bound: float):
        super(ActorNetworkSAC, self).__init__()
        self.action_high_bound = action_high_bound
        # Architecture (adjust complexity as needed)
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, n_actions) # Outputs mean
        self.log_std_layer = nn.Linear(256, n_actions) # Outputs log standard deviation

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Outputs action and its log probability, using reparameterization and tanh squashing.
        Parameters:
        - state (torch.Tensor): Input state.
        Returns:
        - Tuple[torch.Tensor, torch.Tensor]:
            - action: Squashed action sampled from the policy.
            - log_prob: Log probability of the squashed action.
        """
        # Check if state is a single sample and add batch dimension if needed
        add_batch_dim = False
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            add_batch_dim = True
            
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Create Gaussian distribution
        normal_dist = Normal(mean, std)

        # Reparameterization trick: sample pre-squashed action
        # Use rsample() for differentiable sampling
        z = normal_dist.rsample()
        
        # Apply tanh squashing to get bounded action
        action = torch.tanh(z)
        
        # Calculate log-probability with correction for tanh squashing
        # log_prob = log_normal(z) - log(1 - tanh(z)^2)
        log_prob = normal_dist.log_prob(z) - torch.log(1 - action.pow(2) + EPSILON)
        
        # Sum across action dimensions (proper handling of dimensions)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            log_prob = log_prob.sum(keepdim=True)
        
        # Scale action to environment bounds
        action = action * self.action_high_bound
        
        # Remove batch dimension if it was added
        if add_batch_dim:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            
        return action, log_prob

class CriticNetworkSAC(nn.Module):
    """ Twin Q-Value Critic Network for SAC """
    def __init__(self, n_observations: int, n_actions: int):
        super(CriticNetworkSAC, self).__init__()

        # Q1 Architecture
        self.q1_layer1 = nn.Linear(n_observations + n_actions, 256)
        self.q1_layer2 = nn.Linear(256, 256)
        self.q1_output = nn.Linear(256, 1)

        # Q2 Architecture
        self.q2_layer1 = nn.Linear(n_observations + n_actions, 256)
        self.q2_layer2 = nn.Linear(256, 256)
        self.q2_output = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Outputs the Q-values from both internal critics.
        Parameters:
        - state (torch.Tensor): Input state tensor.
        - action (torch.Tensor): Input action tensor.
        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Q1(s, a) and Q2(s, a).
        """
        sa = torch.cat([state, action], dim=1) # Concatenate state and action

        # Q1 forward pass
        q1 = F.relu(self.q1_layer1(sa))
        q1 = F.relu(self.q1_layer2(q1))
        q1 = self.q1_output(q1)

        # Q2 forward pass
        q2 = F.relu(self.q2_layer1(sa))
        q2 = F.relu(self.q2_layer2(q2))
        q2 = self.q2_output(q2)

        return q1, q2

# Define the structure for storing transitions
# Using the same Transition namedtuple as DDPG/DQN
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

# Define the Replay Memory class (Identical to DDPG/DQN version)
class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # Ensure tensors are float32 and on CPU
                processed_args.append(arg.to(torch.float32).cpu())
            elif isinstance(arg, np.ndarray):
                # Convert numpy array to float32 tensor
                processed_args.append(torch.from_numpy(arg).to(torch.float32).cpu())
            elif isinstance(arg, (bool, float, int)):
                # Store scalar values as float32 tensors
                processed_args.append(torch.tensor([arg], dtype=torch.float32))
            else:
                processed_args.append(arg)
        self.memory.append(Transition(*processed_args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

def soft_update(target_net: nn.Module, main_net: nn.Module, tau: float) -> None:
    """ Performs a soft update of the target network parameters. (Identical) """
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

def update_sac(memory: ReplayMemory,
               batch_size: int,
               actor: ActorNetworkSAC,
               critic: CriticNetworkSAC,
               target_critic: CriticNetworkSAC,
               actor_optimizer: optim.Optimizer,
               critic_optimizer: optim.Optimizer,
               log_alpha: torch.Tensor,
               alpha_optimizer: optim.Optimizer,
               target_entropy: float,
               gamma: float,
               tau: float) -> Tuple[float, float, float, float]:
    """
    Performs one SAC update step (critic, actor, alpha).
    """
    # Ensure enough samples are available in memory
    if len(memory) < batch_size:
        return 0.0, 0.0, 0.0, torch.exp(log_alpha.detach()).item()

    # Sample a batch of transitions from memory
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Unpack and move data to the appropriate device with explicit dtype=float32
    state_batch = torch.cat([s.view(1, -1).float() for s in batch.state]).to(device)
    action_batch = torch.cat([a.view(1, -1).float() for a in batch.action]).to(device)
    reward_batch = torch.cat([r.view(1, -1).float() for r in batch.reward]).to(device)
    next_state_batch = torch.cat([s.view(1, -1).float() for s in batch.next_state]).to(device)
    done_batch = torch.cat([d.view(1, -1).float() for d in batch.done]).to(device)

    # --- Critic Update ---
    with torch.no_grad():
        # Get next action and log probability from the current policy
        next_action, next_log_prob = actor(next_state_batch)
        
        # Get target Q values from the target critics
        q1_target_next, q2_target_next = target_critic(next_state_batch, next_action)
        q_target_next = torch.min(q1_target_next, q2_target_next)  # Min of two Q-values
        
        # Calculate the soft target:
        # soft_target = Q_target_next - α * log_prob
        alpha = torch.exp(log_alpha.detach()).float()
        soft_target = q_target_next - alpha * next_log_prob
        
        # Compute the target value for the Bellman equation:
        # y = reward + γ * (1 - done) * soft_target
        y = reward_batch + gamma * (1.0 - done_batch) * soft_target

    # Get current Q estimates from the critic
    q1_current, q2_current = critic(state_batch, action_batch)

    # Calculate critic losses (Mean Squared Error):
    # critic_loss = MSE(Q1_current, y) + MSE(Q2_current, y)
    critic1_loss = F.mse_loss(q1_current, y)
    critic2_loss = F.mse_loss(q2_current, y)
    critic_loss = critic1_loss + critic2_loss

    # Optimize the critic networks
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # --- Actor Update ---
    # Freeze critic gradients to avoid updating them during actor optimization
    for p in critic.parameters():
        p.requires_grad = False

    # Get actions and log probabilities for the current states from the actor
    pi_action, pi_log_prob = actor(state_batch)
    
    # Get Q values for these actions from the critic
    q1_pi, q2_pi = critic(state_batch, pi_action)
    min_q_pi = torch.min(q1_pi, q2_pi)  # Min of two Q-values

    # Calculate actor loss:
    # actor_loss = E[α * log_prob - Q_min]
    actor_loss = (alpha * pi_log_prob - min_q_pi).mean()

    # Optimize the actor network
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Unfreeze critic gradients
    for p in critic.parameters():
        p.requires_grad = True
        
    # --- Alpha (Entropy Temperature) Update ---
    # Calculate alpha loss:
    # alpha_loss = -E[log_alpha * (log_prob + target_entropy)]
    target_entropy_tensor = torch.tensor(target_entropy, dtype=torch.float32, device=device)
    alpha_loss = -(log_alpha * (pi_log_prob.detach().float() + target_entropy_tensor)).mean()

    # Optimize alpha (if auto-tuning is enabled)
    if alpha_optimizer is not None:
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
    
    # Get the current value of alpha
    current_alpha = torch.exp(log_alpha.detach()).item()

    # --- Update Target Networks ---
    # Perform a soft update of the target critic networks:
    # θ_target = τ * θ_main + (1 - τ) * θ_target
    soft_update(target_critic, critic, tau)

    # Return the losses and the current alpha value
    return critic_loss.item(), actor_loss.item(), alpha_loss.item(), current_alpha


# Hyperparameters for SAC on Pendulum-v1
BUFFER_SIZE_SAC = int(1e6)     # Replay buffer capacity
BATCH_SIZE_SAC = 256           # Mini-batch size
GAMMA_SAC = 0.99               # Discount factor
TAU_SAC = 5e-3                 # Soft update factor
LR_SAC = 3e-4                  # Learning rate for actor, critic, and alpha
INITIAL_ALPHA = 0.2            # Initial entropy temperature (or fixed value if not tuning)
AUTO_TUNE_ALPHA = True         # Whether to automatically tune alpha
TARGET_ENTROPY = -float(n_actions_sac) # Heuristic target entropy: -|Action Space Dim|

NUM_EPISODES_SAC = 100         # Number of training episodes
MAX_STEPS_PER_EPISODE_SAC = 200 # Pendulum typically uses 200 steps
START_STEPS = 1000             # Number of initial random steps before training starts
UPDATE_EVERY_SAC = 1           # Perform update after every environment step


if env is None:
    raise RuntimeError("Gymnasium environment 'Pendulum-v1' could not be created.")

# Initialize Networks
actor_sac = ActorNetworkSAC(n_observations_sac, n_actions_sac, action_high_sac).to(device)
critic_sac = CriticNetworkSAC(n_observations_sac, n_actions_sac).to(device)
target_critic_sac = CriticNetworkSAC(n_observations_sac, n_actions_sac).to(device)
target_critic_sac.load_state_dict(critic_sac.state_dict())
# Freeze target critic parameters
for p in target_critic_sac.parameters():
    p.requires_grad = False

# Initialize Optimizers
actor_optimizer_sac = optim.Adam(actor_sac.parameters(), lr=LR_SAC)
critic_optimizer_sac = optim.Adam(critic_sac.parameters(), lr=LR_SAC)

# Initialize Alpha (Entropy Temperature)
# Initialize Alpha (Entropy Temperature) with explicit dtype
if AUTO_TUNE_ALPHA:
    # Learn log_alpha for stability with explicit float32
    log_alpha_sac = torch.tensor(np.log(INITIAL_ALPHA), dtype=torch.float32, requires_grad=True, device=device)
    alpha_optimizer_sac = optim.Adam([log_alpha_sac], lr=LR_SAC)
else:
    log_alpha_sac = torch.tensor(np.log(INITIAL_ALPHA), dtype=torch.float32, requires_grad=False, device=device)
    alpha_optimizer_sac = None # No optimizer needed if alpha is fixed

# Make sure TARGET_ENTROPY is also float32
TARGET_ENTROPY_TENSOR = torch.tensor(-float(n_actions_sac), dtype=torch.float32, device=device)

# Initialize Replay Memory
memory_sac = ReplayMemory(BUFFER_SIZE_SAC)

# Lists for plotting
sac_episode_rewards = []
sac_episode_critic_losses = []
sac_episode_actor_losses = []
sac_episode_alpha_losses = []
sac_episode_alphas = []


print("Starting SAC Training on Pendulum-v1...")

# --- SAC Training Loop ---
total_steps_sac = 0
for i_episode in range(1, NUM_EPISODES_SAC + 1):
    state_np, info = env.reset()
    state = torch.from_numpy(state_np).float().to(device)
    episode_reward = 0
    episode_critic_loss = 0
    episode_actor_loss = 0
    episode_alpha_loss = 0
    num_updates = 0

    for t in range(MAX_STEPS_PER_EPISODE_SAC):
        # --- Action Selection --- 
        if total_steps_sac < START_STEPS:
            # Initial exploration with random actions
            action = env.action_space.sample() # Sample from the environment's action space
            action_tensor = torch.from_numpy(action).float().to(device)
        else:
            # Sample action from the stochastic policy
            actor_sac.eval() # Set to eval mode for consistent sampling
            with torch.no_grad():
                action_tensor, _ = actor_sac(state)
            actor_sac.train() # Back to train mode
            action = action_tensor.cpu().numpy() # Convert to numpy for env.step
            # Action is already scaled by the network
            # Clipping might still be needed if network output + noise slightly exceeds bounds
            action = np.clip(action, action_low_sac, action_high_sac)

        # --- Environment Interaction --- 
        next_state_np, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # --- Store Experience --- 
        # Ensure action stored is a tensor
        action_store_tensor = torch.from_numpy(action if isinstance(action, np.ndarray) else np.array([action])).float()
        memory_sac.push(state, action_store_tensor, reward, next_state_np, done)

        state_np = next_state_np
        state = torch.from_numpy(state_np).float().to(device)
        episode_reward += reward
        total_steps_sac += 1

        # --- Update Networks (if enough steps gathered and buffer full enough) --- 
        if total_steps_sac >= START_STEPS and total_steps_sac % UPDATE_EVERY_SAC == 0:
            if len(memory_sac) > BATCH_SIZE_SAC:
                c_loss, a_loss, alpha_loss, _ = update_sac(
                    memory_sac, BATCH_SIZE_SAC, 
                    actor_sac, critic_sac, target_critic_sac,
                    actor_optimizer_sac, critic_optimizer_sac,
                    log_alpha_sac, alpha_optimizer_sac if AUTO_TUNE_ALPHA else None, 
                    TARGET_ENTROPY if AUTO_TUNE_ALPHA else 0.0,
                    GAMMA_SAC, TAU_SAC
                )
                episode_critic_loss += c_loss
                episode_actor_loss += a_loss
                episode_alpha_loss += alpha_loss
                num_updates += 1

        if done:
            break
            
    # --- End of Episode --- 
    sac_episode_rewards.append(episode_reward)
    sac_episode_critic_losses.append(episode_critic_loss / num_updates if num_updates > 0 else 0)
    sac_episode_actor_losses.append(episode_actor_loss / num_updates if num_updates > 0 else 0)
    sac_episode_alpha_losses.append(episode_alpha_loss / num_updates if num_updates > 0 else 0)
    sac_episode_alphas.append(torch.exp(log_alpha_sac.detach()).item())

    # Print progress
    if i_episode % 10 == 0:
        avg_reward = np.mean(sac_episode_rewards[-10:])
        avg_closs = np.mean(sac_episode_critic_losses[-10:])
        avg_aloss = np.mean(sac_episode_actor_losses[-10:])
        current_alpha = sac_episode_alphas[-1]
        print(f"Ep {i_episode}/{NUM_EPISODES_SAC} | Steps: {total_steps_sac} | Avg Reward: {avg_reward:.2f} | C_Loss: {avg_closs:.4f} | A_Loss: {avg_aloss:.4f} | Alpha: {current_alpha:.4f}")

print("Pendulum-v1 Training Finished (SAC).")

# Plotting results for SAC on Pendulum-v1
plt.figure(figsize=(20, 8))

# Episode Rewards
plt.subplot(2, 3, 1)
plt.plot(sac_episode_rewards)
plt.title('SAC Pendulum: Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
if len(sac_episode_rewards) >= 10:
    rewards_ma_sac = np.convolve(sac_episode_rewards, np.ones(10)/10, mode='valid')
    plt.plot(np.arange(len(rewards_ma_sac)) + 9, rewards_ma_sac, label='10-episode MA', color='orange')
    plt.legend()

# Critic Loss
plt.subplot(2, 3, 2)
plt.plot(sac_episode_critic_losses)
plt.title('SAC Pendulum: Avg Critic Loss / Episode')
plt.xlabel('Episode')
plt.ylabel('Avg MSE Loss')
plt.grid(True)
if len(sac_episode_critic_losses) >= 10:
    closs_ma_sac = np.convolve(sac_episode_critic_losses, np.ones(10)/10, mode='valid')
    plt.plot(np.arange(len(closs_ma_sac)) + 9, closs_ma_sac, label='10-episode MA', color='orange')
    plt.legend()

# Actor Loss
plt.subplot(2, 3, 3)
plt.plot(sac_episode_actor_losses)
plt.title('SAC Pendulum: Avg Actor Loss / Episode')
plt.xlabel('Episode')
plt.ylabel('Avg Loss (alpha*log_pi - Q)')
plt.grid(True)
if len(sac_episode_actor_losses) >= 10:
    aloss_ma_sac = np.convolve(sac_episode_actor_losses, np.ones(10)/10, mode='valid')
    plt.plot(np.arange(len(aloss_ma_sac)) + 9, aloss_ma_sac, label='10-episode MA', color='orange')
    plt.legend()

# Alpha Value
plt.subplot(2, 3, 4)
plt.plot(sac_episode_alphas)
plt.title('SAC Pendulum: Alpha (Entropy Temp) / Episode')
plt.xlabel('Episode')
plt.ylabel('Alpha')
plt.grid(True)

# Alpha Loss (if auto-tuning)
if AUTO_TUNE_ALPHA:
    plt.subplot(2, 3, 5)
    plt.plot(sac_episode_alpha_losses)
    plt.title('SAC Pendulum: Avg Alpha Loss / Episode')
    plt.xlabel('Episode')
    plt.ylabel('Avg Loss')
    plt.grid(True)
    if len(sac_episode_alpha_losses) >= 10:
        alphloss_ma_sac = np.convolve(sac_episode_alpha_losses, np.ones(10)/10, mode='valid')
        plt.plot(np.arange(len(alphloss_ma_sac)) + 9, alphloss_ma_sac, label='10-episode MA', color='orange')
        plt.legend()

plt.tight_layout()
plt.show()

def test_sac_agent(actor_net: ActorNetworkSAC, 
                   env_instance: gym.Env, 
                   num_episodes: int = 5, 
                   render: bool = False, 
                   seed_offset: int = 2000) -> None:
    """
    Tests the trained SAC agent using the mean action (deterministically).
    """
    if env_instance is None:
        print("Environment not available for testing.")
        return
        
    actor_net.eval() # Set actor to evaluation mode
    
    print(f"\n--- Testing SAC Agent ({num_episodes} episodes, Deterministic) ---")
    all_rewards = []
    for i in range(num_episodes):
        state_np, info = env_instance.reset(seed=seed + seed_offset + i)
        state = torch.from_numpy(state_np).float().to(device)
        episode_reward = 0
        done = False
        t = 0
        while not done:
            if render:
                try:
                    env_instance.render()
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Rendering failed: {e}. Disabling render.")
                    render = False
            
            with torch.no_grad():
                # --- Get Deterministic Action (Mean) --- 
                # Forward pass to get mean, ignore sampled action and log_prob
                x = F.relu(actor_net.layer1(state))
                x = F.relu(actor_net.layer2(x))
                mean = actor_net.mean_layer(x)
                action_deterministic = torch.tanh(mean) * actor_net.action_high_bound
                # -----------------------------------------
                action = action_deterministic.cpu().numpy()
            
            # Clipping just in case
            action_clipped = np.clip(action, env_instance.action_space.low, env_instance.action_space.high)
            
            next_state_np, reward, terminated, truncated, _ = env_instance.step(action_clipped)
            done = terminated or truncated
            state = torch.from_numpy(next_state_np).float().to(device)
            episode_reward += reward
            t += 1
        
        print(f"Test Episode {i+1}: Reward = {episode_reward:.2f}, Length = {t}")
        all_rewards.append(episode_reward)
        if render:
             env_instance.close()

    print(f"--- Testing Complete. Average Reward: {np.mean(all_rewards):.2f} ---")

# Run test episodes
test_sac_agent(actor_sac, env, num_episodes=3, render=False) # Set render=True if desired

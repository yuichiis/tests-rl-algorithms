# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple, deque
from itertools import count
from typing import List, Tuple, Dict, Optional, Callable, Any, Union
import copy
import os
import time

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import Gymnasium for the continuous environment
try:
    import gymnasium as gym
except ImportError:
    print("Gymnasium not found. Please install using 'pip install gymnasium' or 'pip install gym[classic_control]'" )
    # You might want to exit or raise an error here if gym is essential
    gym = None # Set gym to None if import fails

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
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
        # Create the environment
        env = gym.make('Pendulum-v1')
        
        # Set seeds for environment reproducibility
        env.reset(seed=seed)
        env.action_space.seed(seed)

        # Get state and action space dimensions
        n_observations_ddpg = env.observation_space.shape[0]
        n_actions_ddpg = env.action_space.shape[0] # DDPG handles continuous actions
        action_low = env.action_space.low[0]
        action_high = env.action_space.high[0]

        print(f"Pendulum Environment:")
        print(f"State Dim: {n_observations_ddpg}")
        print(f"Action Dim: {n_actions_ddpg}")
        print(f"Action Low: {action_low}")
        print(f"Action High: {action_high}")
        
        # Test reset
        obs, info = env.reset()
        print(f"Initial observation: {obs}")
        
    except Exception as e:
        print(f"Error creating Gymnasium environment: {e}")
        # Set dummy values if env creation fails
        n_observations_ddpg = 3
        n_actions_ddpg = 1
        action_low = -2.0
        action_high = 2.0
        env = None # Mark env as unusable
else:
    print("Gymnasium not available. Cannot create Pendulum environment.")
    # Set dummy values
    n_observations_ddpg = 3
    n_actions_ddpg = 1
    action_low = -2.0
    action_high = 2.0
    env = None


# Defineding the Actor Network
class ActorNetwork(nn.Module):
    """ Deterministic Actor Network for DDPG """
    def __init__(self, n_observations: int, n_actions: int, action_high_bound: float):
        super(ActorNetwork, self).__init__()
        self.action_high_bound = action_high_bound
        # Simple MLP architecture
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)
        
        # Initialize final layer weights for smaller initial outputs
        # Often helps in DDPG
        nn.init.uniform_(self.layer3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.layer3.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Maps state to a deterministic action.
        Parameters:
        - state (torch.Tensor): Input state tensor.
        Returns:
        - torch.Tensor: The deterministic action, scaled to environment bounds.
        """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        # Use tanh to bound output between -1 and 1
        action_tanh = torch.tanh(self.layer3(x))
        # Scale to the environment's action bounds
        scaled_action = action_tanh * self.action_high_bound
        return scaled_action
    
# Definding the Critic Network
class CriticNetwork(nn.Module):
    """ Q-Value Critic Network for DDPG """
    def __init__(self, n_observations: int, n_actions: int):
        super(CriticNetwork, self).__init__()
        # Process state separately first
        self.state_layer1 = nn.Linear(n_observations, 256)
        # Combine state features and action in the second layer
        self.combined_layer2 = nn.Linear(256 + n_actions, 256)
        self.output_layer3 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Maps state and action to a Q-value.
        Parameters:
        - state (torch.Tensor): Input state tensor.
        - action (torch.Tensor): Input action tensor.
        Returns:
        - torch.Tensor: The estimated Q(s, a) value.
        """
        state_features = F.relu(self.state_layer1(state))
        # Concatenate state features and action
        combined = torch.cat([state_features, action], dim=1)
        x = F.relu(self.combined_layer2(combined))
        q_value = self.output_layer3(x)
        return q_value


# Definding the Replay Memory
# Define the structure for storing transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

# Define the Replay Memory class
class ReplayMemory(object):
    """ Stores transitions and allows sampling batches. """
    def __init__(self, capacity: int):
        """
        Initialize the Replay Memory.
        Parameters:
        - capacity (int): Maximum number of transitions to store.
        """
        # Use deque for efficient FIFO buffer
        self.memory = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
        """
        Save a transition.
        Parameters:
        - *args: The transition elements (state, action, reward, next_state, done).
                 State/action/reward/next_state should be tensors or easily convertible.
        """
        # Ensure data is stored appropriately (e.g., tensors on CPU)
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                processed_args.append(arg.cpu()) # Store tensors on CPU
            elif isinstance(arg, (bool, float, int)):
                 # Convert bool/float/int to tensors for consistency if needed later,
                 # but storing primitives is fine too.
                 # Let's store primitives for done/reward, tensors for states/actions.
                 processed_args.append(arg) 
            elif isinstance(arg, np.ndarray):
                 processed_args.append(torch.from_numpy(arg).float().cpu()) # Convert numpy arrays
            else:
                 processed_args.append(arg) # Keep others as is
                 
        self.memory.append(Transition(*processed_args))

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample a random batch of transitions from memory.
        Parameters:
        - batch_size (int): The number of transitions to sample.
        Returns:
        - List[Transition]: A list containing the sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """ Return the current size of the memory. """
        return len(self.memory)

# Definding the Exploration Noise
class GaussianNoise:
    """ Simple Gaussian Noise process for exploration. """
    def __init__(self, action_dimension: int, mean: float = 0.0, std_dev: float = 0.1):
        """
        Initialize Gaussian noise.
        Parameters:
        - action_dimension (int): Dimension of the action space.
        - mean (float): Mean of the Gaussian distribution.
        - std_dev (float): Standard deviation of the Gaussian distribution.
        """
        self.action_dim = action_dimension
        self.mean = mean
        self.std_dev = std_dev

    def get_noise(self) -> np.ndarray:
        """ Generate noise. """
        # Generate noise using numpy
        noise = np.random.normal(self.mean, self.std_dev, self.action_dim)
        return noise

    def reset(self) -> None:
        """ Reset noise state (no state for Gaussian noise). """
        pass

# Soft Update Function
def soft_update(target_net: nn.Module, main_net: nn.Module, tau: float) -> None:
    """
    Performs a soft update of the target network parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Parameters:
    - target_net (nn.Module): The target network to be updated.
    - main_net (nn.Module): The main network providing the parameters.
    - tau (float): The soft update factor (τ).
    """
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

# The DDPG Update Step
def update_ddpg(memory: ReplayMemory,
                  batch_size: int,
                  actor: ActorNetwork,
                  critic: CriticNetwork,
                  target_actor: ActorNetwork,
                  target_critic: CriticNetwork,
                  actor_optimizer: optim.Optimizer,
                  critic_optimizer: optim.Optimizer,
                  gamma: float,
                  tau: float) -> Tuple[float, float]:
    """
    Performs one DDPG update step (actor and critic).

    Parameters:
    - memory: The ReplayMemory object.
    - batch_size: The size of the mini-batch to sample.
    - actor, critic: Main networks.
    - target_actor, target_critic: Target networks.
    - actor_optimizer, critic_optimizer: Optimizers.
    - gamma: Discount factor.
    - tau: Soft update factor.

    Returns:
    - Tuple[float, float]: Critic loss and Actor loss for logging.
    """
    # Don't update if buffer doesn't have enough samples
    if len(memory) < batch_size:
        return 0.0, 0.0

    # Sample a batch
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Unpack batch data and move to device
    # Ensure states/next_states are FloatTensors, actions are FloatTensors, rewards/dones are FloatTensors
    state_batch = torch.stack([s for s in batch.state if s is not None]).float().to(device)
    action_batch = torch.stack([a for a in batch.action if a is not None]).float().to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_state_batch = torch.stack([s for s in batch.next_state if s is not None]).float().to(device)
    # Convert boolean 'done' flags to float tensor (1.0 for done, 0.0 for not done)
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    # --- Critic Update --- 
    
    # 1. Calculate target Q-values (y)
    with torch.no_grad(): # Target calculations don't need gradient tracking
        # Get next actions from target actor
        next_actions = target_actor(next_state_batch)
        # Get Q-values for next states/actions from target critic
        target_q_values = target_critic(next_state_batch, next_actions)
        # Compute the target y = r + gamma * Q'_target * (1 - done)
        y = reward_batch + gamma * (1.0 - done_batch) * target_q_values

    # 2. Get current Q-values from main critic
    current_q_values = critic(state_batch, action_batch)

    # 3. Compute Critic loss (MSE)
    critic_loss = F.mse_loss(current_q_values, y)

    # 4. Optimize the Critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    # Optional: Gradient clipping for critic
    # torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_optimizer.step()

    # --- Actor Update --- 

    # 1. Calculate Actor loss (negative mean Q-value for actor's actions)
    # We want to maximize Q(s, mu(s)), so minimize -Q(s, mu(s))
    actor_actions = actor(state_batch)
    q_values_for_actor = critic(state_batch, actor_actions) # Use main critic
    actor_loss = -q_values_for_actor.mean()

    # 2. Optimize the Actor
    actor_optimizer.zero_grad()
    actor_loss.backward()
    # Optional: Gradient clipping for actor
    # torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optimizer.step()

    # --- Update Target Networks --- 
    soft_update(target_critic, critic, tau)
    soft_update(target_actor, actor, tau)

    return critic_loss.item(), actor_loss.item()

# Running the DDPG Algorithm
# Hyperparameters for DDPG on Pendulum-v1
BUFFER_SIZE = int(1e6)     # Replay buffer capacity
BATCH_SIZE = 128           # Mini-batch size for updates
GAMMA_DDPG = 0.99          # Discount factor
TAU = 1e-3                 # Soft update factor for target networks
ACTOR_LR_DDPG = 1e-4       # Learning rate for the actor
CRITIC_LR_DDPG = 1e-3      # Learning rate for the critic (often higher than actor)
WEIGHT_DECAY = 0           # L2 weight decay for critic optimizer (optional)

NOISE_STD_DEV = 0.2        # Standard deviation for Gaussian exploration noise
NOISE_DECAY = 0.999        # Decay factor for noise std dev (optional annealing)
MIN_NOISE_STD_DEV = 0.01   # Minimum noise standard deviation

NUM_EPISODES_DDPG = 100    # Number of training episodes
MAX_STEPS_PER_EPISODE_DDPG = 500 # Max steps per episode in Pendulum
UPDATE_EVERY = 1           # How often to perform update steps (e.g., every step)
NUM_UPDATES = 1            # Number of update steps per UPDATE_EVERY interval

# Initialization
# Ensure environment was created successfully
if env is None:
    raise RuntimeError("Gymnasium environment 'Pendulum-v1' could not be created. Ensure gymnasium is installed.")

# Initialize Networks
actor_ddpg = ActorNetwork(n_observations_ddpg, n_actions_ddpg, action_high).to(device)
critic_ddpg = CriticNetwork(n_observations_ddpg, n_actions_ddpg).to(device)

# Initialize Target Networks (hard copy initially)
target_actor_ddpg = ActorNetwork(n_observations_ddpg, n_actions_ddpg, action_high).to(device)
target_critic_ddpg = CriticNetwork(n_observations_ddpg, n_actions_ddpg).to(device)
target_actor_ddpg.load_state_dict(actor_ddpg.state_dict())
target_critic_ddpg.load_state_dict(critic_ddpg.state_dict())

# Initialize Optimizers
actor_optimizer_ddpg = optim.Adam(actor_ddpg.parameters(), lr=ACTOR_LR_DDPG)
critic_optimizer_ddpg = optim.Adam(critic_ddpg.parameters(), lr=CRITIC_LR_DDPG, weight_decay=WEIGHT_DECAY)

# Initialize Replay Memory
memory_ddpg = ReplayMemory(BUFFER_SIZE)

# Initialize Noise Process
noise = GaussianNoise(n_actions_ddpg, std_dev=NOISE_STD_DEV)
current_noise_std_dev = NOISE_STD_DEV

# Lists for plotting
ddpg_episode_rewards = []
ddpg_episode_actor_losses = []
ddpg_episode_critic_losses = []

# Training Loop
print("Starting DDPG Training on Pendulum-v1...")

# --- DDPG Training Loop ---
total_steps = 0
for i_episode in range(1, NUM_EPISODES_DDPG + 1):
    # Reset environment and noise
    state_np, info = env.reset()
    state = torch.from_numpy(state_np).float().to(device)
    noise.reset()
    noise.std_dev = current_noise_std_dev # Set current noise level
    
    episode_reward = 0
    actor_losses = []
    critic_losses = []

    for t in range(MAX_STEPS_PER_EPISODE_DDPG):
        # --- Action Selection --- 
        actor_ddpg.eval() # Set actor to evaluation mode for action selection
        with torch.no_grad():
            action_deterministic = actor_ddpg(state)
        actor_ddpg.train() # Set back to training mode
        
        # Add exploration noise
        action_noise = noise.get_noise()
        action_noisy = action_deterministic.cpu().numpy() + action_noise # Add noise on CPU
        
        # Clip action to environment bounds
        action_clipped = np.clip(action_noisy, action_low, action_high)

        # --- Environment Interaction --- 
        next_state_np, reward, terminated, truncated, _ = env.step(action_clipped)
        done = terminated or truncated
        
        # --- Store Experience --- 
        # Convert to tensors for storage (store action *before* clipping for critic? No, store clipped action)
        action_tensor = torch.from_numpy(action_clipped).float() # Store the executed action
        next_state_tensor = torch.from_numpy(next_state_np).float()
        # Note: state was already a tensor
        memory_ddpg.push(state, action_tensor, reward, next_state_tensor, done)

        state = next_state_tensor.to(device) # Update state for next loop
        episode_reward += reward
        total_steps += 1

        # --- Update Networks --- 
        if len(memory_ddpg) > BATCH_SIZE and total_steps % UPDATE_EVERY == 0:
            for _ in range(NUM_UPDATES):
                c_loss, a_loss = update_ddpg(
                    memory_ddpg, BATCH_SIZE, 
                    actor_ddpg, critic_ddpg,
                    target_actor_ddpg, target_critic_ddpg,
                    actor_optimizer_ddpg, critic_optimizer_ddpg,
                    GAMMA_DDPG, TAU
                )
                critic_losses.append(c_loss)
                actor_losses.append(a_loss)

        if done:
            break
            
    # --- End of Episode --- 
    ddpg_episode_rewards.append(episode_reward)
    ddpg_episode_actor_losses.append(np.mean(actor_losses) if actor_losses else 0)
    ddpg_episode_critic_losses.append(np.mean(critic_losses) if critic_losses else 0)
    
    # Anneal noise
    current_noise_std_dev = max(MIN_NOISE_STD_DEV, current_noise_std_dev * NOISE_DECAY)
    
    # Print progress
    if i_episode % 10 == 0:
        avg_reward = np.mean(ddpg_episode_rewards[-10:])
        avg_actor_loss = np.mean(ddpg_episode_actor_losses[-10:])
        avg_critic_loss = np.mean(ddpg_episode_critic_losses[-10:])
        print(f"Episode {i_episode}/{NUM_EPISODES_DDPG} | Avg Reward: {avg_reward:.2f} | Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f} | Noise Std: {current_noise_std_dev:.3f}")

print("Pendulum-v1 Training Finished (DDPG).")

# Visualizing the Learning Process
# Plotting results for DDPG on Pendulum-v1
plt.figure(figsize=(18, 4))

# Episode Rewards
plt.subplot(1, 3, 1)
plt.plot(ddpg_episode_rewards)
plt.title('DDPG Pendulum: Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
# Add moving average
if len(ddpg_episode_rewards) >= 10:
    rewards_ma_ddpg = np.convolve(ddpg_episode_rewards, np.ones(10)/10, mode='valid')
    plt.plot(np.arange(len(rewards_ma_ddpg)) + 9, rewards_ma_ddpg, label='10-episode MA', color='orange')
    plt.legend()

# Critic Loss
plt.subplot(1, 3, 2)
plt.plot(ddpg_episode_critic_losses)
plt.title('DDPG Pendulum: Avg Critic Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('MSE Loss')
plt.grid(True)
if len(ddpg_episode_critic_losses) >= 10:
    closs_ma_ddpg = np.convolve(ddpg_episode_critic_losses, np.ones(10)/10, mode='valid')
    plt.plot(np.arange(len(closs_ma_ddpg)) + 9, closs_ma_ddpg, label='10-episode MA', color='orange')
    plt.legend()

# Actor Loss
plt.subplot(1, 3, 3)
plt.plot(ddpg_episode_actor_losses)
plt.title('DDPG Pendulum: Avg Actor Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Avg -Q Value')
plt.grid(True)
if len(ddpg_episode_actor_losses) >= 10:
    aloss_ma_ddpg = np.convolve(ddpg_episode_actor_losses, np.ones(10)/10, mode='valid')
    plt.plot(np.arange(len(aloss_ma_ddpg)) + 9, aloss_ma_ddpg, label='10-episode MA', color='orange')
    plt.legend()

plt.tight_layout()
plt.show()

# Analyzing the Learned Policy (Testing)
def test_ddpg_agent(actor_net: ActorNetwork, 
                    env_instance: gym.Env, 
                    num_episodes: int = 5, 
                    render: bool = False, # Set to True to visualize
                    seed_offset: int = 1000) -> None:
    """
    Tests the trained DDPG agent deterministically.
    
    Parameters:
    - actor_net: The trained actor network.
    - env_instance: An instance of the environment.
    - num_episodes: Number of test episodes to run.
    - render: If True, attempts to render the environment.
    - seed_offset: Offset for seeding test episodes differently from training.
    """
    if env_instance is None:
        print("Environment not available for testing.")
        return
        
    actor_net.eval() # Set actor to evaluation mode (important!)
    
    print(f"\n--- Testing DDPG Agent ({num_episodes} episodes) ---")
    all_rewards = []
    for i in range(num_episodes):
        state_np, info = env_instance.reset(seed=seed + seed_offset + i) # Different seed for testing
        state = torch.from_numpy(state_np).float().to(device)
        episode_reward = 0
        done = False
        t = 0
        while not done:
            if render:
                try:
                    # Try rendering (might require extra setup depending on environment/system)
                    env_instance.render()
                    time.sleep(0.01) # Slow down rendering slightly
                except Exception as e:
                    print(f"Rendering failed: {e}. Disabling render.")
                    render = False # Disable rendering if it fails
            
            with torch.no_grad():
                # Select action deterministically (no noise)
                action = actor_net(state).cpu().numpy()
            
            # Clipping is still important even in testing
            action_clipped = np.clip(action, env_instance.action_space.low, env_instance.action_space.high)
            
            next_state_np, reward, terminated, truncated, _ = env_instance.step(action_clipped)
            done = terminated or truncated
            state = torch.from_numpy(next_state_np).float().to(device)
            episode_reward += reward
            t += 1
        
        print(f"Test Episode {i+1}: Reward = {episode_reward:.2f}, Length = {t}")
        all_rewards.append(episode_reward)
        if render:
             env_instance.close() # Close the render window

    print(f"--- Testing Complete. Average Reward: {np.mean(all_rewards):.2f} ---")

# Run test episodes (ensure env is still available)
env = gym.make('Pendulum-v1', render_mode="human")
test_ddpg_agent(actor_ddpg, env, num_episodes=3, render=False) # Set render=True if you have display setup

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np
import random
import gym
from transformers import BertTokenizer
from src.models.procedural_generator import ProceduralGenerator
from src.training.rl_env import GridGenEnv, MapRNN
from src.models.components import GridVAE
import pickle
import threading
import matplotlib.pyplot as plt
from src.training.train import expand_embedding_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np
import threading
import matplotlib.pyplot as plt
from collections import deque

def extract_latents_for_rnn(vae, grid_sequence):
    """Extract latent vectors from a sequence of grids using the VAE encoder."""
    B, T, H, W = grid_sequence.shape
    flat       = grid_sequence.view(B * T, H, W).long()

    with torch.no_grad():
        x      = vae.embedding(flat)
        x      = vae.pos_enc(x)
        x      = x.permute(0, 3, 1, 2)
        h      = vae.encoder_conv(x)
        mu, _  = torch.chunk(h, 2, dim=1)
        pooled = F.adaptive_avg_pool2d(mu, (1, 1))
        pooled = pooled.view(B, T, -1)

    return pooled

class KeyboardListener:
    """Simple keyboard listener for visualization toggling."""
    def __init__(self, toggle_key="h"):
        self.toggle_key      = toggle_key
        self.visualize       = False
        self.listener_thread = threading.Thread(target=self.listen, daemon=True)
        self.listener_thread.start()

    def listen(self):
        import sys
        import select
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key.lower() == self.toggle_key:
                    self.visualize = not self.visualize


class PPOPolicy(nn.Module):
    """PPO Policy for procedural grid generation."""
    def __init__(self, base_model, context_rnn=None):
        super().__init__()

        self.base_model  = base_model
        self.context_rnn = context_rnn
        self.value_head  = nn.Sequential(
            nn.Linear(base_model.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, grid, mask, text_input, context_embedding=None):
        """Forward pass for policy network."""
        batch_size = grid.size(0)
        
        text_input = {
            k: v.expand(batch_size, -1) if v.size(0) == 1 and batch_size > 1 else v
            for k, v in text_input.items()
        }
        
        with torch.no_grad():
            t = torch.randint(0, self.base_model.timesteps, (batch_size,))
            predictions, latent_features = self.base_model(
                grid, t, text_input, mask, 
                use_blue_noise=False, 
                map_embedding=context_embedding
            )
            logits = F.log_softmax(predictions, dim=1)
            
        return logits, latent_features

    def act(self, grid, mask, text_input, context_embedding=None):
        """Sample actions from the policy."""

        print(f"Grid shape: {grid.shape}, Mask shape: {mask.shape}")
        if grid.dim() == 2:
            grid = grid.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
            
        print(f"Grid shape after unsqueeze: {grid.shape}, Mask shape after unsqueeze: {mask.shape}")
        logits, latent = self.forward(grid, mask, text_input, context_embedding)

        B, C, H, W       = logits.shape
        logits           = logits.permute(0, 2, 3, 1)
        dist             = Categorical(logits=logits)
        actions          = dist.sample()
        log_probs        = dist.log_prob(actions)
        entropy          = dist.entropy()
        mask_bool        = mask.bool().squeeze(0)
        actions_grid     = actions.squeeze(0)
        log_probs_masked = log_probs.squeeze(0)[mask_bool]
        entropy_masked   = entropy.squeeze(0)[mask_bool]
        
        return actions_grid, log_probs_masked, entropy_masked, latent

    def evaluate_actions(self, grid, mask, text_input, actions, context_embedding=None):
        """Evaluate given actions under the current policy."""
        logits, latent = self.forward(grid, mask, text_input, context_embedding)
        
        B, C, H, W       = logits.shape
        logits           = logits.permute(0, 2, 3, 1)
        dist             = Categorical(logits=logits)
        log_probs        = dist.log_prob(actions)
        entropy          = dist.entropy()
        mask_bool        = mask.bool()
        masked_log_probs = []
        masked_entropy   = []
        
        for b in range(B):
            masked_log_probs.append(log_probs[b][mask_bool[b]])
            masked_entropy.append(entropy[b][mask_bool[b]])
            
        return torch.cat(masked_log_probs), torch.cat(masked_entropy), latent

    def get_value(self, latent):
        """Predict state value from latent features."""
        flat_latent = latent.flatten(start_dim=1)
        return self.value_head(flat_latent)


class PPOBuffer:
    """Buffer for storing trajectories collected from the environment."""
    def __init__(self, gamma=0.99, lam=0.95):

        self.states         = []
        self.actions        = []
        self.log_probs      = []
        self.rewards        = []
        self.values         = []
        self.masks          = []
        self.text_inputs    = []
        self.context_states = []
        self.dones          = []
        self.gamma          = gamma
        self.lam            = lam
        
    def store(self, state, action, log_prob, reward, value, mask, text_input, context_state, done):
        """Store a transition."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.masks.append(mask)
        self.text_inputs.append(text_input)
        self.context_states.append(context_state)
        self.dones.append(done)
        
    def compute_advantages_and_returns(self):
        """Compute advantages using GAE and returns for all stored states."""
        advantages = []
        returns = []
        gae = 0
        
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
               
                next_value = 0 if self.dones[i] else self.values[i].item()
            else:
                next_value = self.values[i+1].item()
                
            delta = self.rewards[i] + self.gamma * next_value * (1 - int(self.dones[i])) - self.values[i].item()
            gae   = delta + self.gamma * self.lam * (1 - int(self.dones[i])) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i].item())
            
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return returns, advantages
        
    def get_batch(self):
        """Return all stored data as a batch."""

        states         = self.states
        actions        = self.actions
        masks          = self.masks
        text_inputs    = self.text_inputs[0]
        context_states = self.context_states
        log_probs      = torch.cat(self.log_probs) if self.log_probs else torch.tensor([])
        
        returns, advantages = self.compute_advantages_and_returns()
        
        return states, actions, masks, text_inputs, context_states, log_probs, returns, advantages
        
    def clear(self):
        """Clear buffer."""
        self.states         = []
        self.actions        = []
        self.log_probs      = []
        self.rewards        = []
        self.values         = []
        self.masks          = []
        self.text_inputs    = []
        self.context_states = []
        self.dones          = []


def update_ppo_policy(policy, optimizer, buffer, 
                    epochs=4, clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01):
    """Update policy using PPO algorithm."""
    states, actions, masks, text_input, context_states, old_log_probs, returns, advantages = buffer.get_batch()
    
    total_policy_loss = 0
    total_value_loss  = 0
    total_entropy     = 0
    
    for _ in range(epochs):
        for i in range(len(states)):
            state  = states[i].unsqueeze(0)
            action = actions[i].unsqueeze(0)
            mask   = masks[i].unsqueeze(0)
            adv    = advantages[i]
            ret    = returns[i]
            old_log_prob = old_log_probs[i:i+1] if i < len(old_log_probs) else None
            
            if old_log_prob is None or len(old_log_prob) == 0:
                continue
                
            context_embedding = None
            if policy.context_rnn is not None and i > 0:
                prev_states = [states[j] for j in range(i)]
                if prev_states:
                    prev_states_tensor = torch.stack(prev_states, dim=0).unsqueeze(0)  # [1, prev_steps, H, W]
                    context_features = extract_latents_for_rnn(policy.base_model.vae, prev_states_tensor)
                    context_embedding = policy.context_rnn(context_features)
            
            new_log_probs, entropy, latent = policy.evaluate_actions(
                state, mask, text_input, action, context_embedding
            )
            
            ratio        = torch.exp(new_log_probs - old_log_prob.detach())
            surr1        = ratio * adv
            surr2        = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
            policy_loss  = -torch.min(surr1, surr2).mean()
            value        = policy.get_value(latent)
            value_loss   = F.mse_loss(value.squeeze(-1), ret.unsqueeze(0))
            entropy_loss = entropy.mean()
            loss         = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy_loss.item()
    
    avg_policy_loss = total_policy_loss / (epochs * len(states))
    avg_value_loss  = total_value_loss / (epochs * len(states))
    avg_entropy     = total_entropy / (epochs * len(states))
    
    return avg_policy_loss, avg_value_loss, avg_entropy


def show_grids(grids, rewards=None):
    """Visualize grids with optional rewards."""
    n = len(grids)
    fig, axs = plt.subplots(1, n, figsize=(n * 2, 2))
    if n == 1:
        axs = [axs]
    
    for i, (ax, grid) in enumerate(zip(axs, grids)):
        ax.imshow(grid.cpu().numpy(), cmap='tab20')
        ax.axis('off')
        if rewards:
            ax.set_title(f"Reward: {rewards[i]:.4f}")
    
    plt.tight_layout()
    plt.show()


def show_full_map(full_grid, grid_H, grid_W, side, reward=None):
    """Visualize the complete map with grid lines."""
    plt.figure(figsize=(8, 8))
    plt.imshow(full_grid.cpu().numpy(), cmap='tab20')
    
    for i in range(1, side):
        plt.axhline(y=i * grid_H - 0.5, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=i * grid_W - 0.5, color='k', linestyle='-', alpha=0.3)
    
    plt.title(f"Complete Map (Reward: {reward:.4f})" if reward is not None else "Complete Map")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def train_ppo(base_model, tokenizer, label_to_idx, idx_to_label, 
              episodes=1000, grid_H=64, grid_W=64, grids_per_episode=9,
              learning_rate=1e-4, gamma=0.99, lam=0.95):
    """Train a PPO agent for procedural grid generation."""

    env             = GridGenEnv(base_model, tokenizer, label_to_idx, idx_to_label, grid_H, grid_W, grids_per_episode)
    side            = int(grids_per_episode ** 0.5)
    context_rnn     = MapRNN()
    policy          = PPOPolicy(base_model, context_rnn)
    optimizer       = Adam(list(policy.parameters()) + list(context_rnn.parameters()), lr=learning_rate)
    buffer          = PPOBuffer(gamma=gamma, lam=lam)
    listener        = KeyboardListener()
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        grid, mask, text_input = env.reset()
        
        episode_reward = 0
        step           = 0
        done           = False
        episode_grids  = []
        step_rewards   = []
        full_grid      = None
        context_states = []
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        print(f"Description: {env.description}")
        
        while not done:
            context_embedding = None
            if step > 0 and context_rnn is not None:
                if len(episode_grids) > 0:
                    prev_grids = torch.stack(episode_grids, dim=0).unsqueeze(0) 
                    context_features = extract_latents_for_rnn(base_model.vae, prev_grids)
                    context_embedding = context_rnn(context_features)
                    context_states.append(context_embedding)
            
            action_grid, log_probs, entropy, latent = policy.act(grid, mask, text_input, context_embedding)            
            value = policy.get_value(latent)
            (next_grid, next_mask, next_text_input), reward, done, _ = env.step(action_grid.cpu().numpy())
            
            buffer.store(
                grid.squeeze(0),
                action_grid,
                log_probs,
                reward,
                value,
                mask.squeeze(0),
                text_input,
                context_embedding,
                done
            )
            
            grid, mask, text_input = next_grid, next_mask, next_text_input
            episode_reward        += reward
            step                  += 1
            
            episode_grids.append(action_grid)
            step_rewards.append(reward)
            
            print(f"  Step {step}/{grids_per_episode}: Reward = {reward:.4f}")
            
            if listener.visualize:
                print("\n[Visualization paused] Press 'H' again to resume training...")
                if done: 
                    full_grid = env._stitch_history()
                    show_full_map(full_grid, grid_H, grid_W, side, episode_reward)
                else:  
                    show_grids(episode_grids, step_rewards)
                while listener.visualize:
                    pass
        
        if len(buffer.rewards) > 0:
            update_ppo_policy(policy, optimizer, buffer)
            buffer.clear()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
            
        print(f"Episode {episode + 1} complete. Total Reward: {episode_reward:.4f}")
        
        if episode % 50 == 0:
            torch.save({
                'policy_state_dict'     : policy.state_dict(),
                'context_rnn_state_dict': context_rnn.state_dict(),
                'optimizer_state_dict'  : optimizer.state_dict(),
                'episode'               : episode,
                'reward'                : episode_reward
            }, f"ppo_checkpoint_ep{episode}.pth")
    
    torch.save({
        'policy_state_dict'     : policy.state_dict(),
        'context_rnn_state_dict': context_rnn.state_dict(),
        'optimizer_state_dict'  : optimizer.state_dict()
    }, "ppo_final_model.pth")
    
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('PPO Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.savefig('ppo_learning_curve.png')
    plt.show()
    
    return policy, context_rnn

if __name__ == "__main__":

    device = "cpu"

    with open("data/output/specific_processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    all_labels                                      = {cell for item in data for row in item["grid_labels"] for cell in row if cell is not None}
    label_to_idx                                    = {label: i for i, label in enumerate(sorted(all_labels))}
    idx_to_label                                    = {i: label for label, i in label_to_idx.items()}

    grid_H, grid_W                                  = 64, 64
    embed_dim, num_heads, ff_dim                    = 64, 4, 128
    time_embed_dim, diffusion_hidden_dim, timesteps = 128, 256, 1000
    tokenizer                                       = BertTokenizer.from_pretrained("bert-base-uncased")
    base_model                                      = ProceduralGenerator(embed_dim, num_heads, ff_dim, time_embed_dim,
                            diffusion_hidden_dim, (grid_H, grid_W), len(label_to_idx), grid_H, grid_W, 32, timesteps)

    vae_ckpt   = "data/checkpoints/vae_best_model.pth"
    vae        = GridVAE(len(label_to_idx), 64, 64, 64, latent_channels=32)
    state_dict = torch.load(vae_ckpt, map_location="cpu") 
    vae.load_state_dict(state_dict)

    vae.embedding.embedding = expand_embedding_weights(vae.embedding.embedding, len(label_to_idx) + 1)
    vae.to("cpu")
    
    checkpoint_path = "data/checkpoints/procedural_generator_model_epoch.pth"
    state_dict      = torch.load(checkpoint_path, map_location="cpu") 
    base_model.vae  = vae
    base_model.load_state_dict(state_dict, strict=False)
    vae.eval()
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    for param in base_model.vae.parameters():
        param.requires_grad = False

    print("✅ VAE frozen: base_model.vae will not be updated during PPO training.")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    ppo_model = train_ppo(base_model, tokenizer, label_to_idx, idx_to_label, episodes=1000)
    torch.save(ppo_model.state_dict(), "ppo_trained.pth")

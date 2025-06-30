import torch
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from tqdm import tqdm

# ─── (1) Your existing imports/models/dataset definitions ────────────────────
from src.dataset.grid_dataset import GridDataset
from src.models.components import GridVAE
from src.generators.proc_generator import ProcGenerator
from sklearn.metrics import f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"


with open("data/output/g_maps.pkl", "rb") as f:
    data = pickle.load(f)

labels = ProcGenerator.labels
label_to_idx = { i: i for i in range(len(labels)) }

random.shuffle(data)
train_size = int(0.7 * len(data))
val_size   = int(0.15 * len(data))
test_size  = len(data) - train_size - val_size

train_data = data[:train_size]
val_data   = data[train_size:train_size + val_size]
test_data  = data[train_size + val_size:]

train_dataset = GridDataset(train_data, label_to_idx, mode="full")
val_dataset   = GridDataset(val_data,   label_to_idx, mode="full")
test_dataset  = GridDataset(test_data,  label_to_idx, mode="full")

# ─── (3) Create three DataLoaders ─────────────────────────────────────────────

batch_size = 2

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# ─── (4) Instantiate your VAE, optimizer, loss criterion ──────────────────────

grid_H, grid_W    = 64, 64
embed_dim         = 32
latent_channels   = 32
learning_rate     = 1e-4
num_epochs        = 11

vae = GridVAE(
    num_labels= train_dataset.vocab_size,
    embed_dim=embed_dim,
    grid_H=grid_H,
    grid_W=grid_W,
    latent_channels=latent_channels
).to(device)

optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# ─── NEW: Initialize history lists ────────────────────────────────────────────
train_recon_history = []
train_kl_history    = []
train_total_history = []
train_acc_history   = []
train_f1_history    = []

val_recon_history   = []
val_kl_history      = []
val_total_history   = []
val_acc_history     = []
val_f1_history      = []

best_loss  = float("inf")
best_epoch = -1
target_weight = 0.01
kl_weight = 0.0

for epoch in range(num_epochs):
    # ─── TRAINING PHASE ────────────────────────────────────────────────────────
    vae.train()
    kl_weight = min(kl_weight + 0.001, target_weight)

    # Accumulators for this epoch (TRAIN)
    epoch_train_recon_sum = 0.0
    epoch_train_kl_sum    = 0.0
    epoch_train_total_sum = 0.0

    epoch_train_correct_tokens = 0
    epoch_train_total_tokens   = 0

    # For F1, we will accumulate all predictions/labels for the entire epoch,
    # then call sklearn.metrics.f1_score once at the end of the epoch.
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for (full_grids, _, _, _) in progress_bar:
        full_grids = full_grids.to(device)  # shape: [batch, H, W]

        optimizer.zero_grad()
        logits, mu, logvar = vae(full_grids)
        # logits: [batch, vocab_size, H, W]; full_grids: [batch, H, W]

        # 1) Reconstruction loss
        loss_recon = criterion(logits, full_grids)

        # 2) KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / full_grids.size(0)

        # 3) Total loss
        loss = loss_recon + kl_weight * kl_div

        # BACKPROP
        loss.backward()
        optimizer.step()

        # ─── Compute token-level accuracy (TRAIN BATCH) ───────────────────
        preds = logits.argmax(dim=1)  # [batch, H, W]
        correct = (preds == full_grids).sum().item()
        total_tokens_batch = full_grids.numel()  # batch * H * W

        # Append these batch’s preds/labels (flattened) for F1
        all_preds.append(preds.view(-1).cpu())
        all_labels.append(full_grids.view(-1).cpu())

        # Accumulate sums
        epoch_train_recon_sum += loss_recon.item()
        epoch_train_kl_sum    += kl_div.item()
        epoch_train_total_sum += loss.item()

        epoch_train_correct_tokens += correct
        epoch_train_total_tokens   += total_tokens_batch

        progress_bar.set_postfix(
            train_loss=f"{loss.item():.4f}",
            recon=f"{loss_recon.item():.4f}",
            kl=f"{kl_div.item():.4f}"
        )

    # ─── End of train‐batches for this epoch ───────────────────────────────────
    avg_train_recon = epoch_train_recon_sum / len(train_dataloader)
    avg_train_kl    = epoch_train_kl_sum    / len(train_dataloader)
    avg_train_total = epoch_train_total_sum / len(train_dataloader)
    train_acc       = epoch_train_correct_tokens / epoch_train_total_tokens

    # Compute F1 over the entire epoch (TRAIN)
    all_preds_tensor = torch.cat(all_preds).numpy()
    all_labels_tensor = torch.cat(all_labels).numpy()
    train_f1 = f1_score(all_labels_tensor, all_preds_tensor, average="macro")

    # Save to history
    train_recon_history.append(avg_train_recon)
    train_kl_history.append(avg_train_kl)
    train_total_history.append(avg_train_total)
    train_acc_history.append(train_acc)
    train_f1_history.append(train_f1)

    print(f"📘 Epoch {epoch+1}, Train ▶ Recon={avg_train_recon:.4f}, KL={avg_train_kl:.4f}, Total={avg_train_total:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")

    # ─── VALIDATION PHASE ──────────────────────────────────────────────────────
    vae.eval()
    with torch.no_grad():
        epoch_val_recon_sum = 0.0
        epoch_val_kl_sum    = 0.0
        epoch_val_total_sum = 0.0

        epoch_val_correct_tokens = 0
        epoch_val_total_tokens   = 0

        val_all_preds = []
        val_all_labels = []

        for (full_grids_val, _, _, _) in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Valid]", leave=False):
            full_grids_val = full_grids_val.to(device)
            logits_val, mu_val, logvar_val = vae(full_grids_val)

            loss_recon_val = criterion(logits_val, full_grids_val)
            kl_div_val = -0.5 * torch.sum(
                1 + logvar_val - mu_val.pow(2) - logvar_val.exp()
            ) / full_grids_val.size(0)
            loss_val = loss_recon_val + kl_weight * kl_div_val

            # TOKEN‑LEVEL ACCURACY (VALID)
            preds_val = logits_val.argmax(dim=1)
            correct_val = (preds_val == full_grids_val).sum().item()
            total_tokens_val = full_grids_val.numel()

            val_all_preds.append(preds_val.view(-1).cpu())
            val_all_labels.append(full_grids_val.view(-1).cpu())

            epoch_val_recon_sum += loss_recon_val.item()
            epoch_val_kl_sum    += kl_div_val.item()
            epoch_val_total_sum += loss_val.item()

            epoch_val_correct_tokens += correct_val
            epoch_val_total_tokens   += total_tokens_val

        # ─── End of validation batches ──────────────────────────────────────────
        avg_val_recon = epoch_val_recon_sum / len(val_dataloader)
        avg_val_kl    = epoch_val_kl_sum    / len(val_dataloader)
        avg_val_total = epoch_val_total_sum / len(val_dataloader)
        val_acc       = epoch_val_correct_tokens / epoch_val_total_tokens

        # COMPUTE F1 (VALID)
        val_preds_tensor = torch.cat(val_all_preds).numpy()
        val_labels_tensor = torch.cat(val_all_labels).numpy()
        val_f1 = f1_score(val_labels_tensor, val_preds_tensor, average="macro")

        # Save to history
        val_recon_history.append(avg_val_recon)
        val_kl_history.append(avg_val_kl)
        val_total_history.append(avg_val_total)
        val_acc_history.append(val_acc)
        val_f1_history.append(val_f1)

        print(f"🟦 Epoch {epoch+1}, Val ▶ Recon={avg_val_recon:.4f}, KL={avg_val_kl:.4f}, Total={avg_val_total:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

        # Save best‐model if validation total loss improved
        if avg_val_total < best_loss:
            best_loss  = avg_val_total
            best_epoch = epoch + 1
            best_path  = "vae_best_model_epoch.pth"
            torch.save(vae.state_dict(), best_path)
            print(f"✅ New best model saved to {best_path}  (epoch {best_epoch})")

last_model_path = "vae_last_epoch.pth"
torch.save(vae.state_dict(), last_model_path)
print(f"💾 Last-trained model (epoch {num_epochs}) saved to {last_model_path}")

# ─── (6) FINAL TEST LOOP (same as before, if you like) ─────────────────────────
vae.eval()
total_test_loss = 0.0

with torch.no_grad():
    for (full_grids_test, _, _, _) in tqdm(test_dataloader, desc="Final [Test]", leave=False):
        full_grids_test = full_grids_test.to(device)
        logits_test, mu_test, logvar_test = vae(full_grids_test)

        loss_recon_test = criterion(logits_test, full_grids_test)
        kl_div_test     = -0.5 * torch.sum(
            1 + logvar_test - mu_test.pow(2) - logvar_test.exp()
        ) / full_grids_test.size(0)

        loss_test = loss_recon_test + kl_weight * kl_div_test
        total_test_loss += loss_test.item()

avg_test_loss = total_test_loss / len(test_dataloader)
print(f"🟩 Final Test Loss: {avg_test_loss:.4f}")

history = {
    "train_recon": train_recon_history,
    "train_kl":    train_kl_history,
    "train_total": train_total_history,
    "train_acc":   train_acc_history,
    "train_f1":    train_f1_history,

    "val_recon":   val_recon_history,
    "val_kl":      val_kl_history,
    "val_total":   val_total_history,
    "val_acc":     val_acc_history,
    "val_f1":      val_f1_history,
}

with open("vae_training_history.pkl", "wb") as fp:
    pickle.dump(history, fp)

print("✅ Saved training history to vae_training_history.pkl")
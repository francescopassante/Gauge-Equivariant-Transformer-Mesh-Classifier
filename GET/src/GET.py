import time

import GEBlocks
import GEData
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


class GETClassifier(nn.Module):
    def __init__(self, N, channels, out_classes):
        super().__init__()
        self.N = N
        self.channels = channels

        self.local_to_regular = GEBlocks.GELocalToRegularLinearBlock(N, channels)
        self.self_attention1 = GEBlocks.GESelfAttentionBlock(N, channels)
        self.self_attention2 = GEBlocks.GESelfAttentionBlock(N, channels)
        self.group_pool = GEBlocks.GEGroupPooling()
        self.global_average_pool = GEBlocks.GEGlobalAveragePooling()
        self.fc = nn.Linear(channels, out_classes)

    def forward(self, x, neighbors, mask, parallel_transport_matrices, rel_pos_u):
        # x: (1, N_v, 3)

        # Local to regular transformation
        x0 = self.local_to_regular(x)  # (N_v, channels, N)
        x0 = torch.relu(x0)  # (N_v, channels, N)

        # Self-attention
        x = self.self_attention1(
            x0, neighbors, mask, parallel_transport_matrices, rel_pos_u
        )  # (N_v, in_channels, N)
        x = torch.relu(x)  # (N_v, in_channels, N)

        x = self.self_attention2(
            x, neighbors, mask, parallel_transport_matrices, rel_pos_u
        )  # (N_v, in_channels, N)
        x = x + x0  # Residual connection

        x = self.group_pool(x)  # (N_v, in_channels)
        x = self.global_average_pool(x)  # (in_channels)
        return self.fc(x)  # (classes)


def train(
    model, dataloader, optimizer, criterion, device, epochs=1, accumulation_steps=16
):
    model.train()
    loss_hist = []

    for epoch in range(epochs):
        total_loss = 0.0

        # Start with zeroed gradients for the first accumulation block
        optimizer.zero_grad()

        for i, mesh in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            x = mesh["x"].to(device).squeeze(0)
            neighbors = mesh["neighbors"].to(device).squeeze(0)
            mask = mesh["mask"].to(device).squeeze(0)
            parallel_transport_matrices = (
                mesh["parallel_transport"].to(device).squeeze(0)
            )
            rel_pos_u = mesh["rel_pos"].to(device).squeeze(0)
            labels = mesh["label"].to(device).long().squeeze(0)

            # Forward
            outputs = model(x, neighbors, mask, parallel_transport_matrices, rel_pos_u)
            raw_loss = criterion(outputs.unsqueeze(0), labels.unsqueeze(0))

            # Scale the loss for gradient accumulation
            scaled_loss = raw_loss / accumulation_steps
            scaled_loss.backward()

            # Accumulate the raw (unscaled) loss for logging
            total_loss += raw_loss.item()

            # Step and zero gradients at the accumulation boundary (or final batch)
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        # Average epoch loss (uses unscaled batch losses)
        epoch_loss = total_loss / len(dataloader)
        loss_hist.append(epoch_loss)

        # Save checkpoint with meaningful loss value (epoch average)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }
        save_path = "checkpoint.pth"
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path} (epoch_loss={epoch_loss:.6f})")

    return loss_hist


def load_data(mesh_directory, labels_file, N, train_percent):
    # Esempio di utilizzo:
    full_dataset = GEData.MeshDataset(mesh_directory, labels_file, N)

    # 2. Calcola le dimensioni
    train_size = int(train_percent * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # 3. Esegui lo split casuale
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 4. Crea i DataLoader separati
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    device = "mps"

    train_loader, test_loader = load_data(
        mesh_directory="../data/SHREC11/processed/",
        labels_file="../data/SHREC11/classes.txt",
        N=9,
        train_percent=0.1,
    )

    print(len(train_loader), len(test_loader))

    model = GETClassifier(N=9, channels=12, out_classes=30).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.005)

    loss_hist = train(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=2,
    )

    print(loss_hist)

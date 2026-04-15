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
        # x: (B, N_v, 3)

        # Local to regular transformation
        x0 = self.local_to_regular(x)  # (B, N_v, channels, N)
        x0 = torch.relu(x0)  # (B, N_v, channels, N)

        # Self-attention
        x = self.self_attention1(
            x0, neighbors, mask, parallel_transport_matrices, rel_pos_u
        )  # (B, N_v, in_channels, N)
        x = torch.relu(x)  # (B, N_v, in_channels, N)

        x = self.self_attention2(
            x, neighbors, mask, parallel_transport_matrices, rel_pos_u
        )  # (B, N_v, in_channels, N)
        x = x + x0  # Residual connection

        x = self.group_pool(x)  # (B, N_v, in_channels)
        x = self.global_average_pool(x)  # (B, in_channels)
        return self.fc(x)


def train(model, dataloader, optimizer, criterion, device, epochs=1):
    model.train()
    loss_hist = []
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        for mesh in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x, neighbors, mask, parallel_transport_matrices, rel_pos_u, labels = mesh
            x = mesh["x"].to(device).squeeze(0)
            neighbors = mesh["neighbors"].to(device).squeeze(0)
            mask = mesh["mask"].to(device).squeeze(0)
            parallel_transport_matrices = (
                mesh["parallel_transport"].to(device).squeeze(0)
            )
            rel_pos_u = mesh["rel_pos"].to(device).squeeze(0)
            labels = mesh["label"].to(device).squeeze(0)

            optimizer.zero_grad()
            outputs = model(x, neighbors, mask, parallel_transport_matrices, rel_pos_u)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        epoch_loss = total_loss / len(dataloader)
        loss_hist.append(epoch_loss)
        print(
            f"Epoch {epoch + 1}/{epochs} finished in {epoch_time:.2f}s - avg loss: {epoch_loss:.4f}"
        )

        save_path = f"model_final_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved model state_dict to {save_path}")

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


if __name__ == "__main__"
    device = "mps"

    train_loader, test_loader = load_data(
        mesh_directory="../data/processed/",
        labels_file="../data/For_evaluation/test.cla",
        N=9,
        train_percent=0.8,
    )

    print(len(train_loader), len(test_loader))

    model = GETClassifier(N=9, channels=12, out_classes=30).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.005)

    train(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=1,
    )

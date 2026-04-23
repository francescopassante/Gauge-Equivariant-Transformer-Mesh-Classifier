import GEBlocks
import GEData
import GEUtils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


class GETClassifier(nn.Module):
    def __init__(self, N, channels, heads, out_classes, num_blocks=1):
        super().__init__()
        self.local_to_regular = GEBlocks.GELocalToRegularLinearBlock(N, channels)

        self.blocks = nn.ModuleList([
            GEBlocks.GEResNetBlock(N, channels, heads) for _ in range(num_blocks)
        ])

        self.group_pool = GEBlocks.GEGroupPooling()
        self.global_average_pool = GEBlocks.GEGlobalAveragePooling()
        self.fc = nn.Linear(channels, out_classes)

    def forward(self, x, neighbors, mask, parallel_transport_matrices, rel_pos_u):
        x = torch.relu(self.local_to_regular(x))

        for block in self.blocks:
            x = block(x, neighbors, mask, parallel_transport_matrices, rel_pos_u)

        x = self.group_pool(x)
        x = self.global_average_pool(x)
        return self.fc(x)


def validate(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataloader.
    Returns (avg_loss, accuracy_percent).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for mesh in dataloader:
            x = mesh["x"].to(device).squeeze(0)
            neighbors = mesh["neighbors"].to(device).squeeze(0)
            mask = mesh["mask"].to(device).squeeze(0)
            parallel_transport_matrices = (
                mesh["parallel_transport_matrices"].to(device).squeeze(0)
            )
            rel_pos_u = mesh["rel_pos"].to(device).squeeze(0)
            labels = mesh["label"].to(device).long().squeeze(0)

            outputs = model(x, neighbors, mask, parallel_transport_matrices, rel_pos_u)
            loss = criterion(outputs.unsqueeze(0), labels.unsqueeze(0))
            total_loss += loss.item()

            pred = outputs.argmax()
            correct += (pred == labels).item()
            total += 1

    model.train()
    return total_loss / len(dataloader), 100.0 * correct / total


def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs=1,
    accumulation_steps=16,
    val_loader=None,
    patience=10,
    min_delta=1e-4,
    start_epoch=0,
    train_filenumbers=None,
    test_filenumbers=None,
):
    """
    Train the model.

    Args:
        val_loader:         Optional validation DataLoader. When provided, validation
                            loss is computed after each epoch and early stopping is
                            applied when it stops improving.
        patience:           Number of epochs without improvement before stopping.
        min_delta:          Minimum decrease in val loss to count as an improvement.
        start_epoch:        Epoch to start from (use when resuming from a checkpoint).
        train_filenumbers:  List of file numbers in the training set. Saved in every
                            checkpoint so the session can be reproduced.
        test_filenumbers:   Complementary list for the test set.

    Returns:
        (train_loss_hist, val_loss_hist)  — val_loss_hist is empty when val_loader is None.
    """
    model.train()
    train_loss_hist = []
    val_loss_hist = []

    best_val_loss = float("inf")
    patience_counter = 0

    N = model.local_to_regular.N

    def _make_checkpoint(epoch, epoch_loss, val_loss=None):
        return {
            "epoch": epoch,
            "N": N,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "train_filenumbers": train_filenumbers,
            "test_filenumbers": test_filenumbers,
        }

    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0.0
        optimizer.zero_grad()

        for i, mesh in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            x = mesh["x"].to(device).squeeze(0)
            neighbors = mesh["neighbors"].to(device).squeeze(0)
            mask = mesh["mask"].to(device).squeeze(0)
            parallel_transport_matrices = (
                mesh["parallel_transport_matrices"].to(device).squeeze(0)
            )
            rel_pos_u = mesh["rel_pos"].to(device).squeeze(0)
            labels = mesh["label"].to(device).long().squeeze(0)

            outputs = model(x, neighbors, mask, parallel_transport_matrices, rel_pos_u)
            raw_loss = criterion(outputs.unsqueeze(0), labels.unsqueeze(0))

            if torch.isnan(raw_loss):
                print("NAN LOSS")

            scaled_loss = raw_loss / accumulation_steps
            scaled_loss.backward()

            total_loss += raw_loss.item()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        epoch_loss = total_loss / len(dataloader)
        train_loss_hist.append(epoch_loss)
        scheduler.step()

        # Validation
        val_loss = None
        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            val_loss_hist.append(val_loss)
            print(
                f"Epoch {epoch + 1}: train_loss={epoch_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%"
            )

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(_make_checkpoint(epoch, epoch_loss, val_loss), "checkpoint_best.pth")
                print(f"  -> New best val loss {best_val_loss:.4f}, saved checkpoint_best.pth")
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{patience})")
        else:
            print(f"Epoch {epoch + 1}: train_loss={epoch_loss:.4f}")

        # Always save the latest checkpoint so training can be resumed
        torch.save(_make_checkpoint(epoch, epoch_loss, val_loss), "checkpoint.pth")

        if val_loader is not None and patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    return train_loss_hist, val_loss_hist


def load_data(mesh_directory, labels_file, N, train_percent, device="cpu"):
    """
    Create train/test DataLoaders from a random split of the dataset.

    Returns:
        (train_loader, test_loader, train_filenumbers, test_filenumbers)

    The filenumber lists can be saved and passed to load_data_from_session()
    to reproduce the exact same split later.
    """
    full_dataset = GEData.MeshDataset(mesh_directory, labels_file, N)

    train_size = int(train_percent * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_subset, test_subset = random_split(full_dataset, [train_size, test_size])

    train_filenumbers = [full_dataset.filenumbers[i] for i in train_subset.indices]
    test_filenumbers = [full_dataset.filenumbers[i] for i in test_subset.indices]

    train_loader = DataLoader(
        train_subset, batch_size=1, shuffle=True,
        num_workers=2, pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_subset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=(device == "cuda"),
    )

    return train_loader, test_loader, train_filenumbers, test_filenumbers


def load_data_from_session(checkpoint_path, mesh_directory, labels_file, device="cpu"):
    """
    Recreate the exact train/test DataLoaders from a saved checkpoint.

    The checkpoint must contain 'train_filenumbers', 'test_filenumbers', and 'N'
    (all saved automatically by train()).

    Returns:
        (train_loader, test_loader, checkpoint)

    The returned checkpoint dict can be used to restore the model, optimizer,
    and scheduler states and resume training from start_epoch=checkpoint["epoch"]+1.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    train_filenumbers = checkpoint["train_filenumbers"]
    test_filenumbers = checkpoint["test_filenumbers"]
    N = checkpoint["N"]

    train_dataset = GEData.MeshDataset(
        mesh_directory, labels_file, N, filenumbers=train_filenumbers
    )
    test_dataset = GEData.MeshDataset(
        mesh_directory, labels_file, N, filenumbers=test_filenumbers
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=2, pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=(device == "cuda"),
    )

    return train_loader, test_loader, checkpoint


def check_gauge_invariance(data, N, channels, heads):

    x = data["x"].squeeze(0)
    neighbors = data["neighbors"].squeeze(0)
    parallel_transport_matrices = data["parallel_transport_matrices"].squeeze(0)

    rel_pos_u = data["rel_pos"].squeeze(0)
    mask = data["mask"].squeeze(0)

    # generate a tensor of N_v random angles of the form 2pik/N k integer:
    angles = torch.randint(0, N, (x.shape[0],)) * (2 * torch.pi / N)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # fmt: off
    rot_mat_3d = torch.stack([
        torch.stack([cos, -sin, torch.zeros_like(cos)], dim=-1),
        torch.stack([sin,  cos, torch.zeros_like(cos)], dim=-1),
        torch.stack([torch.zeros_like(cos), torch.zeros_like(cos), torch.ones_like(cos)], dim=-1)
    ], dim=-2)
    # fmt: on

    x_rot = torch.einsum("vij,vj->vi", rot_mat_3d, x)
    rot_rel_pos_u = torch.einsum("vij,vnj->vni", rot_mat_3d[:, :2, :2], rel_pos_u)
    # The parallel transport angles transform as: new_theta_nv = theta_nv + random_angle_v - random_angle_n
    r2r = GEUtils.RegularToRegular(N)
    rot_parallel_transport_matrices = (
        parallel_transport_matrices
        @ r2r.extended_regular_representation(angles.unsqueeze(-1))
        @ r2r.extended_regular_representation(-angles[neighbors])
    )

    model = GETClassifier(N=N, channels=channels, heads=heads, out_classes=30)
    print(
        x.shape,
        neighbors.shape,
        mask.shape,
        rot_parallel_transport_matrices.shape,
        rel_pos_u.shape,
    )

    normal_output = model(x, neighbors, mask, parallel_transport_matrices, rel_pos_u)
    rot_output = model(
        x_rot, neighbors, mask, rot_parallel_transport_matrices, rot_rel_pos_u
    )

    return normal_output, rot_output


if __name__ == "__main__":
    device = "mps"

    train_loader, test_loader, train_filenumbers, test_filenumbers = load_data(
        mesh_directory="../data/SHREC11_200NEIGH/processed/",
        labels_file="../data/SHREC11_200NEIGH/classes.txt",
        N=9,
        train_percent=0.7,
        device=device,
    )

    print(f"Train: {len(train_loader)} samples, Test: {len(test_loader)} samples")

    model = GETClassifier(N=9, channels=12, heads=2, out_classes=30, num_blocks=1).to(device)
    # model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params} parameters")

    train_loss_hist, val_loss_hist = train(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=100,
        accumulation_steps=4,
        val_loader=test_loader,
        patience=15,
        train_filenumbers=train_filenumbers,
        test_filenumbers=test_filenumbers,
    )

    print("train_loss_hist:", train_loss_hist)
    print("val_loss_hist:  ", val_loss_hist)

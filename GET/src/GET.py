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


def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs=1,
    accumulation_steps=16,
):
    model.train()
    loss_hist = []

    for epoch in range(epochs):
        total_loss = 0.0

        optimizer.zero_grad()

        for i, mesh in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            x = mesh["x"].to(device).squeeze(0)
            neighbors = mesh["neighbors"].to(device).squeeze(0)
            mask = mesh["mask"].to(device).squeeze(0)
            parallel_transport_matrices = (
                mesh["parallel_transport_matrices"].to(device).squeeze(0)
            )
            rel_pos_u = mesh["rel_pos"].to(device).squeeze(0)
            labels = mesh["label"].to(device).long().squeeze(0)

            # Forward
            outputs = model(x, neighbors, mask, parallel_transport_matrices, rel_pos_u)
            raw_loss = criterion(outputs.unsqueeze(0), labels.unsqueeze(0))

            if torch.isnan(raw_loss):
                print("NAN LOSS")

            # Scale the loss for gradient accumulation
            scaled_loss = raw_loss / accumulation_steps
            scaled_loss.backward()

            # Accumulate the raw loss for logging
            total_loss += raw_loss.item()

            # Step and zero gradients after accumulation_steps steps or at the end
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Average epoch loss
        epoch_loss = total_loss / len(dataloader)
        print("epoch_loss: ", epoch_loss)
        loss_hist.append(epoch_loss)
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Save checkpoint with meaningful loss value (epoch average)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            }
            save_path = "checkpoint.pth"
            torch.save(checkpoint, save_path)
            # print(f"Saved checkpoint to {save_path} (epoch_loss={epoch_loss:.6f})")

    return loss_hist


def load_data(mesh_directory, labels_file, N, train_percent, device="cpu"):
    full_dataset = GEData.MeshDataset(mesh_directory, labels_file, N)

    train_size = int(train_percent * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=2, pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=(device == "cuda"),
    )

    return train_loader, test_loader


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

    train_loader, test_loader = load_data(
        mesh_directory="../data/SHREC11_200NEIGH/processed/",
        labels_file="../data/SHREC11_200NEIGH/classes.txt",
        N=9,
        train_percent=0.002,
        device=device,
    )

    print(len(train_loader), len(test_loader))

    model = GETClassifier(N=9, channels=12, heads=2, out_classes=30, num_blocks=1).to(device)
    # model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # scheduler to divide by 10 the lr at the 41st epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Model parameters:
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params} parameters")

    loss_hist = train(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=40,
        accumulation_steps=1,
    )

    print(loss_hist)

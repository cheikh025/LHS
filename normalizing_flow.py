import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional


class CouplingLayer(nn.Module):
    """
    Affine coupling layer for RealNVP-style normalizing flow.
    Splits input into two parts and applies invertible affine transformation.
    """
    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor, dropout: float = 0.0):
        super().__init__()
        self.register_buffer('mask', mask)

        # Scale network: outputs log(scale) for numerical stability
        scale_layers = [
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.2),
        ]
        if dropout > 0:
            scale_layers.append(nn.Dropout(dropout))
        scale_layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        ])
        if dropout > 0:
            scale_layers.append(nn.Dropout(dropout))
        scale_layers.extend([
            nn.Linear(hidden_dim, dim),
            nn.Tanh()  # Constrain to [-1, 1], then scale
        ])
        self.scale_net = nn.Sequential(*scale_layers)

        # Translation network
        translate_layers = [
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.2),
        ]
        if dropout > 0:
            translate_layers.append(nn.Dropout(dropout))
        translate_layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        ])
        if dropout > 0:
            translate_layers.append(nn.Dropout(dropout))
        translate_layers.append(nn.Linear(hidden_dim, dim))
        self.translate_net = nn.Sequential(*translate_layers)

        # Scaling factor for tanh output (prevents extreme scales)
        self.scale_factor = 0.8

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: z -> u

        Args:
            z: Input tensor [batch_size, dim]

        Returns:
            u: Transformed tensor [batch_size, dim]
            log_det: Log determinant of Jacobian [batch_size]
        """
        # Split based on mask
        z_masked = z * self.mask

        # Compute scale (log scale) and translation
        log_s = self.scale_net(z_masked) * self.scale_factor
        t = self.translate_net(z_masked)

        # Apply affine transformation to unmasked part
        # u = mask * z + (1-mask) * (z * exp(s) + t)
        u = z_masked + (1 - self.mask) * (z * torch.exp(log_s) + t)

        # Log determinant: sum of log scales for unmasked dimensions
        log_det = torch.sum((1 - self.mask) * log_s, dim=1)

        return u, log_det

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """
        Inverse: u -> z

        Args:
            u: Input tensor [batch_size, dim]

        Returns:
            z: Inverse transformed tensor [batch_size, dim]
        """
        # Split based on mask
        u_masked = u * self.mask

        # Compute scale and translation (same as forward)
        log_s = self.scale_net(u_masked) * self.scale_factor
        t = self.translate_net(u_masked)

        # Invert affine transformation
        # z = mask * u + (1-mask) * ((u - t) * exp(-s))
        z = u_masked + (1 - self.mask) * ((u - t) * torch.exp(-log_s))

        return z


class ActNorm(nn.Module):
    """
    Activation Normalization layer (invertible batch normalization).
    Normalizes activations using learned scale and bias.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.initialized = False

    def initialize(self, x: torch.Tensor):
        """Data-dependent initialization"""
        with torch.no_grad():
            # Compute statistics
            if x.shape[0] > 1:
                mean = x.mean(dim=0)
                std = x.std(dim=0, unbiased=False) + 1e-6
            else:
                # For single sample, just use zeros
                mean = torch.zeros(self.dim, device=x.device)
                std = torch.ones(self.dim, device=x.device)

            # Initialize to normalize the data
            self.bias.data = -mean
            self.log_scale.data = -torch.log(std)
            self.initialized = True

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: z -> u

        Args:
            z: Input tensor [batch_size, dim]

        Returns:
            u: Normalized tensor [batch_size, dim]
            log_det: Log determinant [batch_size]
        """
        if not self.initialized:
            self.initialize(z)

        # Apply: u = (z + bias) * exp(log_scale)
        u = (z + self.bias) * torch.exp(self.log_scale)

        # Log determinant: sum of log_scale (constant for all samples)
        log_det = torch.sum(self.log_scale).expand(z.shape[0])

        return u, log_det

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """
        Inverse: u -> z

        Args:
            u: Input tensor [batch_size, dim]

        Returns:
            z: Denormalized tensor [batch_size, dim]
        """
        # Invert: z = u * exp(-log_scale) - bias
        z = u * torch.exp(-self.log_scale) - self.bias
        return z


class NormalizingFlow(nn.Module):
    """
    RealNVP-based Normalizing Flow: F_φ: R^d -> R^d

    Maps latent codes z to standard Gaussian base distribution u ~ N(0, I)
    """
    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 128, dropout: float = 0.0):
        """
        Args:
            dim: Dimension of latent space
            num_layers: Number of coupling layers (default: 4 for small datasets)
            hidden_dim: Hidden dimension for coupling networks (default: 128 for small datasets)
            dropout: Dropout probability for regularization (default: 0.0)
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Build alternating coupling layers
        self.layers = nn.ModuleList()
        self.actnorms = nn.ModuleList()

        for i in range(num_layers):
            # Create alternating masks (checkerboard pattern)
            mask = torch.zeros(dim)
            if i % 2 == 0:
                # Mask first half
                mask[:dim//2] = 1
            else:
                # Mask second half
                mask[dim//2:] = 1

            # Add ActNorm layer for stability
            self.actnorms.append(ActNorm(dim))

            # Add coupling layer with dropout
            self.layers.append(CouplingLayer(dim, hidden_dim, mask, dropout=dropout))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: z -> u
        Maps latent codes to base distribution N(0, I)

        Args:
            z: Latent codes [batch_size, dim]

        Returns:
            u: Base distribution samples [batch_size, dim]
            log_det_jacobian: Log |det(∂F_φ/∂z)| [batch_size]
        """
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        u = z

        for actnorm, coupling in zip(self.actnorms, self.layers):
            # Apply ActNorm
            u, log_det_actnorm = actnorm(u)
            log_det_sum += log_det_actnorm

            # Apply coupling layer
            u, log_det_coupling = coupling(u)
            log_det_sum += log_det_coupling

        return u, log_det_sum

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: u -> z
        Maps base distribution to latent codes

        Args:
            u: Base distribution samples [batch_size, dim]

        Returns:
            z: Latent codes [batch_size, dim]
        """
        z = u

        # Apply inverse transformations in reverse order
        for actnorm, coupling in reversed(list(zip(self.actnorms, self.layers))):
            z = coupling.inverse(z)
            z = actnorm.inverse(z)

        return z


def compute_flow_loss(flow_model: NormalizingFlow, z_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute normalizing flow loss (negative log-likelihood).

    L_flow(φ) = -1/M Σ[log N(F_φ(z_j); 0, I) + log|det(∂F_φ(z_j)/∂z_j)|]

    Args:
        flow_model: Normalizing flow model
        z_batch: Batch of latent codes [batch_size, dim]

    Returns:
        loss: Negative log-likelihood (scalar)
    """
    # Forward pass through flow
    u, log_det = flow_model(z_batch)

    # Log probability under standard Gaussian N(0, I)
    # log p(u) = -0.5 * ||u||^2 - 0.5 * d * log(2π)
    dim = u.shape[1]
    log_prob_gaussian = -0.5 * torch.sum(u ** 2, dim=1) - 0.5 * dim * np.log(2 * np.pi)

    # Change of variables: log p(z) = log p(u) + log|det(∂F/∂z)|
    log_prob = log_prob_gaussian + log_det

    # Negative log-likelihood (we want to maximize log_prob, so minimize -log_prob)
    nll = -torch.mean(log_prob)

    return nll


def train_flow(
    flow_model: NormalizingFlow,
    program_db,
    batch_size: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = 'cuda',
    verbose: bool = True,
    checkpoint_path: Optional[str] = None
) -> NormalizingFlow:
    """
    Train the normalizing flow model on program embeddings.

    Args:
        flow_model: NormalizingFlow model instance
        program_db: ProgramDatabase containing programs with z embeddings
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization strength
        device: Device to train on ('cuda' or 'cpu')
        verbose: Print training progress
        checkpoint_path: Path to save model checkpoints (optional)

    Returns:
        Trained flow model
    """
    # Extract embeddings from database
    if len(program_db) == 0:
        raise ValueError("ProgramDatabase is empty. Cannot train flow model.")

    z_list = program_db.df['z'].tolist()
    z_array = np.stack(z_list).astype(np.float32)
    z_tensor = torch.tensor(z_array, dtype=torch.float32)

    print(f"Training flow on {len(z_tensor)} program embeddings of dimension {z_tensor.shape[1]}")

    # Create dataloader
    dataset = TensorDataset(z_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Move model to device
    flow_model = flow_model.to(device)
    flow_model.train()

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler: reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for (z_batch,) in dataloader:
            z_batch = z_batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Compute flow loss
            loss = compute_flow_loss(flow_model, z_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=5.0)

            # Update parameters
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Average loss for epoch
        avg_loss = epoch_loss / max(num_batches, 1)

        # Update learning rate
        scheduler.step(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': flow_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)

        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {current_lr:.6f}")

    if verbose:
        print(f"\nTraining complete! Best loss: {best_loss:.4f}")

    return flow_model


def load_flow_checkpoint(flow_model: NormalizingFlow, checkpoint_path: str, device: str = 'cuda'):
    """
    Load flow model from checkpoint.

    Args:
        flow_model: NormalizingFlow model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded flow model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    flow_model.load_state_dict(checkpoint['model_state_dict'])
    flow_model = flow_model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    return flow_model


def sample_from_flow(flow_model: NormalizingFlow, num_samples: int, device: str = 'cuda') -> torch.Tensor:
    """
    Sample new latent codes by sampling from N(0, I) and applying inverse flow.

    Args:
        flow_model: Trained normalizing flow
        num_samples: Number of samples to generate
        device: Device to use

    Returns:
        z_samples: Sampled latent codes [num_samples, dim]
    """
    flow_model.eval()
    with torch.no_grad():
        # Sample from standard Gaussian
        u = torch.randn(num_samples, flow_model.dim, device=device)

        # Apply inverse flow
        z = flow_model.inverse(u)

    return z


if __name__ == '__main__':
    # Example usage
    print("Normalizing Flow Implementation")
    print("=" * 50)

    # Test with random data
    dim = 768  # Typical embedding dimension
    batch_size = 32

    # Create model
    flow = NormalizingFlow(dim=dim, num_layers=4, hidden_dim=128, dropout=0.1)
    print(f"Created flow model with {sum(p.numel() for p in flow.parameters()):,} parameters")
    print(f"Dropout: 0.1 (active in training mode only)")
    print(f"Architecture: 4 layers, hidden_dim=128 (suitable for small datasets)")

    # Test forward and inverse
    z = torch.randn(batch_size, dim)
    u, log_det = flow(z)
    z_reconstructed = flow.inverse(u)

    reconstruction_error = torch.mean((z - z_reconstructed) ** 2)
    print(f"Reconstruction error: {reconstruction_error.item():.6f}")
    print(f"Log det shape: {log_det.shape}")
    print(f"Mean log det: {log_det.mean().item():.4f}")

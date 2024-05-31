import torch
import torch.nn as nn
import torch.nn.functional as F

class ChebyshevKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, grid_size=5, grid_range=(-1, 1)):
        super(ChebyshevKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.grid_size = grid_size
        self.grid_range = grid_range

        # Initialize coefficients for Chebyshev polynomials
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.xavier_normal_(self.cheby_coeffs)
        
        # Register grid points as a buffer
        self.register_buffer("grid", torch.linspace(grid_range[0], grid_range[1], grid_size))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def chebyshev_polynomials(self, x):
        # Compute Chebyshev polynomials at the grid points
        T = [torch.ones_like(x), x]
        for n in range(2, self.degree + 1):
            T.append(2 * x * T[n - 1] - T[n - 2])
        return torch.stack(T, dim=-1)

    def forward(self, x):
        # Normalize input to [-1, 1]
        x = x.view(-1, self.input_dim)
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x = 2 * (x - x_min) / (x_max - x_min) - 1

        # Evaluate Chebyshev polynomials at the normalized input
        T = self.chebyshev_polynomials(x)

        # Compute the output using polynomial coefficients
        y = torch.einsum("bij,ioj->bo", T, self.cheby_coeffs)
        y = y.view(-1, self.output_dim)
        return y

    def update_grid(self, x):
        # Sort data and select grid points adaptively
        x_sorted, _ = x.sort(dim=0)
        grid_adaptive = x_sorted[
            torch.linspace(
                0, x.size(0) - 1, self.grid_size, dtype=torch.long, device=x.device
            )
        ]

        # Uniform grid step
        uniform_step = (x_sorted[-1] - x_sorted[0]) / (self.grid_size - 1)
        grid_uniform = (
            torch.arange(self.grid_size, dtype=torch.float32, device=x.device)
            * uniform_step
            + x_sorted[0]
        )

        # Update grid as a combination of uniform and adaptive grids
        epsilon = 0.02
        self.grid.copy_((1 - epsilon) * grid_adaptive + epsilon * grid_uniform)

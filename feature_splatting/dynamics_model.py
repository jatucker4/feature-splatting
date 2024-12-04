import torch
import torch.nn as nn

# class VelocityTransformer(nn.Module):
#     def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=512, dropout=0.1):
#         super(VelocityTransformer, self).__init__()
#         # Transformer encoder to model Gaussian dynamics
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.transformer_encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True
#             # activation='relu'
#         )
#         self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_encoder_layers).to(self.device)
#         self.d_model = d_model
#         # Fully connected layer to output velocity (vx, vy, vz) for each point and delta t
#         self.fc_out = nn.Linear(d_model, 3).to(self.device)  # Output (vx, vy, vz, delta_t)
#         # torch.backends.cuda.enable_mem_efficient_sdp(False)
#         # torch.backends.cuda.enable_flash_sdp(False)
#         # torch.backends.cuda.enable_math_sdp(True)

#     def forward(self, feature_embeddings, positions, timesteps):
#         # clip_embeddings: Tensor of shape (N, D_clip) representing CLIP feature embeddings
#         # dinov2_embeddings: Tensor of shape (N, D_dinov2) representing Dinov2 feature embeddings
#         # positions: Tensor of shape (N, 3) representing the current positions (x, y, z) of each point
#         # timesteps: Tensor of shape (N, 1) representing the current time step for each point
#         timesteps = torch.ones((positions.shape[0], 1))*timesteps
#         timesteps = timesteps.to("cuda:0")
#         # Concatenate CLIP embeddings, Dinov2 embeddings, positions, and timesteps to form the input
#         transformer_input = torch.cat((feature_embeddings, positions, timesteps), dim=-1)  # Shape: (N, D_clip + D_dinov2 + 3 + 1)

#         # Project concatenated input to match the expected transformer input dimension
#         transformer_input = nn.Linear(transformer_input.shape[-1], self.d_model, device="cuda:0")(transformer_input)  # Shape: (N, D)
        
#         # Add batch dimension for transformer input
#         transformer_input = transformer_input.unsqueeze(0)  # Shape: (1, N, D)
#         # Pass through the transformer encoder
#         transformer_output = self.transformer_encoder(transformer_input)  # Shape: (1, N, D)
        
#         # Remove batch dimension and pass through fully connected layer to predict velocity and delta t
#         transformer_output = transformer_output.squeeze(0)  # Shape: (N, D)
#         velocities_delta_t = self.fc_out(transformer_output)  # Shape: (N, 4)
        
#         # # Split the output into velocities and delta_t
#         # velocities = velocities_delta_t[:, :3]  # Shape: (N, 3)
#         # delta_t = velocities_delta_t[:, 3]  # Shape: (N,)
        
#         # return velocities, delta_t
#         return velocities_delta_t

class VelocityTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=512, dropout=0.1):
        super(VelocityTransformer, self).__init__()
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transformer encoder to model Gaussian dynamics
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_encoder_layers
        )
        self.d_model = d_model

        # Linear layer to project input to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)

        # Fully connected layer to output velocity (vx, vy, vz) for each point
        self.fc_out = nn.Linear(d_model, 3)

        # Move the entire model to the device
        self.to(self.device)

    def forward(self, feature_embeddings, positions, timesteps):
        """
        Args:
            feature_embeddings: Tensor of shape (N, D_feature)
            positions: Tensor of shape (N, 3)
            timesteps: Scalar or tensor of shape (N,) or (N, 1)

        Returns:
            velocities: Tensor of shape (N, 3)
        """
        # Ensure all inputs are on the correct device
        feature_embeddings = feature_embeddings.to(self.device)
        positions = positions.to(self.device)

        # Handle timesteps input
        if not torch.is_tensor(timesteps):
            # If timesteps is a scalar (e.g., float or int), convert it to a tensor
            timesteps = torch.tensor(timesteps, device=self.device)
        else:
            timesteps = timesteps.to(self.device)

        if timesteps.dim() == 0:
            # Scalar timesteps; create a tensor of shape (N, 1) filled with the scalar value
            timesteps = timesteps * torch.ones((positions.shape[0], 1), device=self.device)
        elif timesteps.dim() == 1:
            # Timesteps of shape (N,); unsqueeze to (N, 1)
            timesteps = timesteps.unsqueeze(1)
        elif timesteps.dim() == 2:
            # Timesteps already in shape (N, 1)
            pass
        else:
            raise ValueError(f"timesteps has unexpected dimensions: {timesteps.shape}")
        # Concatenate inputs
        transformer_input = torch.cat((feature_embeddings, positions, timesteps), dim=-1)  # Shape: (N, input_dim)

        # Project concatenated input to match the expected transformer input dimension
        transformer_input = self.input_projection(transformer_input)  # Shape: (N, d_model)

        # Add batch dimension for transformer input
        transformer_input = transformer_input.unsqueeze(0)  # Shape: (1, N, d_model)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(transformer_input)  # Shape: (1, N, d_model)

        # Remove batch dimension and pass through fully connected layer to predict velocity
        transformer_output = transformer_output.squeeze(0)  # Shape: (N, d_model)
        velocities = self.fc_out(transformer_output)  # Shape: (N, 3)

        return velocities
    
# Example usage
def example_usage():
    # Define the transformer model
    # Calculate input dimension
    D_feature = 512+384
    input_dim = D_feature + 3 + 1 
    transformer_model = VelocityTransformer(input_dim=input_dim, d_model=256, nhead=8, num_encoder_layers=6)
    
    # Create dummy CLIP embeddings, Dinov2 embeddings, positions, and timesteps (e.g., for 10 points with different embedding sizes)
    feature_embeddings = torch.randn(10, 512+384).to("cuda:0")  # Shape: (N, D_dinov2), assuming Dinov2 embedding size is 384
    positions = torch.randn(10, 3).to("cuda:0")   # Shape: (N, 3)
    timesteps = 1  # Shape: (N, 1)
    
    # Forward pass to get velocities and delta_t
    velocities = transformer_model(feature_embeddings, positions, timesteps)  # Shapes: (N, 3), (N,)
    print(velocities)

if __name__ == "__main__":
    example_usage()



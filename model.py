import jax
import jax.numpy as jnp
import flax.linen as nn

class GFlowNetModel(nn.Module):
    dim: int
    side: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        
        # Output structure: 
        # [0 : 2*dim+1] -> Forward Logits (Directions + Exit)
        # [2*dim+1 : 4*dim+2] -> Backward Logits
        # [-1] -> Log State Flow (log F(s))
        return nn.Dense(4 * self.dim + 3)(x)
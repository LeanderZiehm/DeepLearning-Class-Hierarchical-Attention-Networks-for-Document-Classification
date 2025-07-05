class BaseModel:
    """Base class for all models."""

    def __init__(self):
        self.model_config = {}

    def forward(self, *args, **kwargs):
        """Forward pass for the model. To be implemented by subclasses."""
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def save(self, filepath):
        """Save the model state to a file."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """Load the model state from a file."""
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict)

    def get_model_config(self):
        """Return the model configuration."""
        return self.model_config
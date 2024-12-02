import torch
from transformers import TrainerCallback
from transformers.utils import logging

# Initialize the Hugging Face logger
logger = logging.get_logger(__name__)


class EMACallback(TrainerCallback):
    def __init__(self, ema_decay=0.999):
        self.ema_decay = ema_decay  # Decay rate for EMA
        self.ema_weights = {}  # Store EMA weights
        self.original_weights = {}  # Store original weights when swapping

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Initialize EMA weights to be the same as the model's initial weights
        self.ema_weights = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        logger.info("EMA weights initialized.")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Update EMA weights after each step
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Update EMA parameter
                    self.ema_weights[name].mul_(self.ema_decay).add_(
                        param, alpha=1 - self.ema_decay
                    )
        logger.debug(f"EMA weights updated at step {state.global_step}.")

    def apply_ema_weights(self, model):
        # Swap the model's weights with the EMA weights
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.ema_weights:
                param.data.copy_(self.ema_weights[name])
        logger.info("EMA weights applied to the model.")

    def save_original_weights(self, model):
        # Store a copy of the original weights to restore later
        self.original_weights = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        logger.debug("Original model weights saved.")

    def restore_original_weights(self, model):
        # Restore the model's original weights after evaluation
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.original_weights:
                param.data.copy_(self.original_weights[name])
        logger.info("Original model weights restored.")

    def on_save(self, args, state, control, model=None, **kwargs):
        # Save EMA weights instead of current weights
        logger.info("Saving model with EMA weights.")
        self.apply_ema_weights(model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Apply EMA weights at the end of training
        logger.info("Applying EMA weights to the final model at the end of training.")
        self.apply_ema_weights(model)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # Before evaluation: swap to EMA weights
        logger.info("Applying EMA weights for evaluation.")
        self.save_original_weights(model)
        self.apply_ema_weights(model)

    def on_evaluate_end(self, args, state, control, model=None, **kwargs):
        # After evaluation: restore original weights
        logger.info("Restoring original weights after evaluation.")
        self.restore_original_weights(model)

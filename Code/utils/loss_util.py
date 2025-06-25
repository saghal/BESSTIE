"""
Filename: weighted_loss_trainer.py

This module provides a custom Trainer class for training Transformer-based models with a weighted
Cross Entropy loss function, along with a utility function to compute evaluation metrics.

Features:
    - WeightedCELossTrainer: A subclass of the Hugging Face Trainer that supports custom negative and positive
      class weights for the cross entropy loss. This is useful when dealing with imbalanced datasets.
    - compute_metrics: A helper function that calculates standard evaluation metrics such as precision,
      recall, f1-score, and accuracy using the evaluate library.

Prerequisites:
    - PyTorch and Torch.nn modules
    - Transformers library from Hugging Face
    - The evaluate package from Hugging Face
    - NumPy
    - Pydantic (for data validation)
    - typing (for type hints)

Usage:
    To use the custom trainer, instantiate WeightedCELossTrainer with additional keyword arguments 'neg_weight'
    and 'pos_weight'. Then, use it in place of the standard Trainer for model training. The compute_metrics
    function can be passed to the Trainer as the metric computation callback.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, field_validator
from transformers import PreTrainedModel, Trainer


class WeightedLossConfig(BaseModel):
    """
    Configuration for weighted loss calculation.

    Attributes:
        neg_weight (float): Weight to apply to the negative class (class 0).
        pos_weight (float): Weight to apply to the positive class (class 1).
    """

    neg_weight: float = Field(1.0, description="Weight for negative class (class 0)")
    pos_weight: float = Field(1.0, description="Weight for positive class (class 1)")

    @field_validator("neg_weight", "pos_weight")
    @classmethod
    def validate_weights(cls, v: float) -> float:
        """Validate that weights are positive numbers."""
        if v <= 0:
            raise ValueError("Weights must be positive numbers")
        return v


class WeightedCELossTrainer(Trainer):
    """
    A custom Trainer class that computes a weighted Cross Entropy loss for Transformer models.

    This class extends the Hugging Face Trainer by allowing the specification of weights for the negative
    and positive classes. These weights are then incorporated into the loss calculation using PyTorch's
    CrossEntropyLoss during model training.

    Args:
        model (PreTrainedModel): The model to train.
        args (Any): Training arguments.
        neg_weight (float, optional): The weight to apply to the negative class in the loss function.
        pos_weight (float, optional): The weight to apply to the positive class in the loss function.
        **kwargs: Additional keyword arguments for the base Trainer.

    Attributes:
        loss_config (WeightedLossConfig): Configuration for the weighted loss function.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: Any,
        neg_weight: float = 1.0,
        pos_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(model=model, args=args, **kwargs)
        self.loss_config = WeightedLossConfig(
            neg_weight=neg_weight, pos_weight=pos_weight
        )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Computes the weighted Cross Entropy loss for a given batch of inputs.

        The loss is calculated by first moving the input tensors and labels to the correct device,
        obtaining the logits from the model, and then applying a weighted CrossEntropyLoss using the
        specified negative and positive weights.

        Args:
            model: The model being trained.
            inputs: A dictionary containing model inputs including the key "labels".
            return_outputs: If True, returns a tuple (loss, outputs); otherwise, returns the loss only.

        Returns:
            The computed loss, or a tuple (loss, outputs) when return_outputs is True.
        """
        # Determine the device (handling DataParallel models if necessary)
        device = (
            model.module.device if isinstance(model, nn.DataParallel) else model.device
        )

        # Move inputs to the appropriate device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract labels and remove them from the inputs passed to the model
        labels = inputs.pop("labels")

        # Run the model and get outputs (which include logits)
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if not logits.requires_grad:
            logits.requires_grad_()

        # Setup the loss function with the configured weights
        weight_tensor = torch.tensor(
            [self.loss_config.neg_weight, self.loss_config.pos_weight],
            device=device,
            dtype=logits.dtype,
        )

        loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)

        # Determine the number of labels from the model configuration
        if isinstance(model, nn.parallel.DistributedDataParallel):
            num_labels = model.module.config.num_labels
        else:
            num_labels = model.config.num_labels

        # Compute the loss
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


@dataclass
class EvaluationMetrics:
    """
    Container for evaluation metrics in classification tasks.

    Attributes:
        precision (float): The precision score.
        recall (float): The recall score.
        f1 (float): The F1 score.
        accuracy (float): The accuracy score.
    """

    precision: float
    recall: float
    f1: float
    accuracy: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to a dictionary format."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1-score": self.f1,
            "accuracy": self.accuracy,
        }


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Computes evaluation metrics (precision, recall, F1-score, and accuracy) based on model predictions.

    This utility function leverages the evaluate library to calculate common metrics for classification tasks.
    It takes the logits and true labels as input, determines the predicted class by taking the argmax over logits,
    and then computes each metric. The output is a dictionary mapping metric names to their computed values.

    Args:
        eval_pred: A tuple (logits, labels) where:
            - logits: The output predictions from the model.
            - labels: The ground truth labels.

    Returns:
        A dictionary containing precision, recall, F1 score, and accuracy metrics.
    """
    # Load metric objects from the evaluate library
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = precision_metric.compute(predictions=predictions, references=labels)[
        "precision"
    ]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]

    metrics = EvaluationMetrics(
        precision=precision, recall=recall, f1=f1, accuracy=accuracy
    )

    return metrics.to_dict()

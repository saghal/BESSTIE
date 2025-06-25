"""
Module: benchmark.py

This module provides functionality for training transformer-based models for various NLP tasks
including sequence classification and causal language modeling. It supports advanced training
techniques such as:

1. Quantization: 4-bit training via BitsAndBytes
2. Parameter-Efficient Fine-Tuning (PEFT): Using LoRA (Low-Rank Adaptation)
3. Weighted loss for imbalanced classification tasks
4. Early stopping and model checkpointing
5. HuggingFace Hub integration for model sharing

The module handles the full training and evaluation pipeline:
- Loading and preprocessing data
- Configuring models with appropriate hyperparameters
- Training with appropriate trainers (WeightedCELossTrainer for classification, SFTTrainer for language modeling)
- Evaluating models on test data
- Saving metrics and predictions
- Pushing models to HuggingFace Hub (optional)

Prerequisites:
- HuggingFace Transformers, Datasets, and PEFT libraries
- BitsAndBytes for quantization support
- PyTorch
- Custom utility modules for data loading and weighted loss computation

Usage:
    python3 benchmark.py [arguments]

    or

    python3 benchmark.py config.json
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers.utils import send_example_telemetry
from trl import SFTTrainer
from utils.arguments import (
    DataArguments,
    LoraArguments,
    QuantArguments,
    TransformerArguments,
)
from utils.data_util import get_data
from utils.loss_util import WeightedCELossTrainer

# Configure logging
logger = logging.getLogger(__name__)

# Environment variables for better performance and stability
os.environ["WANDB_MODE"] = "disabled"  # Disable Weights & Biases logging
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"  # Disable NCCL monitoring
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronous CUDA operations


def find_all_linear_names(model: PreTrainedModel) -> List[str]:
    """
    Finds all linear layer names in a model that can be targeted for LoRA adaptation.

    This function scans through the model's modules to identify 4-bit linear layers
    that are suitable for LoRA adaptation, collecting their names for targeting.

    Args:
        model: A pre-trained model to scan for linear layers.

    Returns:
        A list of module names that can be targeted for LoRA fine-tuning.
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # LM head should not be adapted with LoRA in 16-bit models
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def set_random_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility across different libraries.

    Args:
        seed: The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def predict_sequence_classification(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_length: int = 256,
) -> List[int]:
    """
    Predicts class labels for a list of texts using a sequence classification model.

    Args:
        texts: List of text inputs to classify.
        model: A fine-tuned sequence classification model.
        tokenizer: The tokenizer corresponding to the model.
        device: The device to run inference on.
        max_length: Maximum sequence length for tokenization.

    Returns:
        A list of predicted class labels.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for text in tqdm(texts, desc="Predicting"):
            # Tokenize and prepare inputs
            inputs = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Get model predictions
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            logits = outputs.logits
            probs = softmax(logits, dim=1).cpu().numpy()
            predictions.append(np.argmax(probs))

    # Free up GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return predictions


def predict_causal_lm(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    delimiter: str,
) -> List[str]:
    """
    Generates text completions for causal language models and extracts answers.

    Args:
        texts: List of prompts to generate completions for.
        model: A fine-tuned causal language model.
        tokenizer: The tokenizer corresponding to the model.
        device: The device to run inference on.
        delimiter: String delimiter to extract the answer from the generated text.

    Returns:
        A list of extracted answers from generated text.
    """
    answers = []

    for text in tqdm(texts, desc="Generating"):
        # Tokenize the input text
        encodeds = tokenizer(text, return_tensors="pt")
        model_inputs = encodeds.to(device)

        # Generate output tokens
        generated_ids = model.generate(
            input_ids=encodeds["input_ids"],
            attention_mask=encodeds["attention_mask"],
            max_length=encodeds["input_ids"].shape[1] + 1,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode generated tokens and extract answer
        decoded = tokenizer.batch_decode(generated_ids)
        try:
            word = decoded[0].split(delimiter, 1)[1].strip()
        except IndexError:
            # Handle case where delimiter is not found
            logger.warning(f"Delimiter '{delimiter}' not found in output: {decoded[0]}")
            word = decoded[0].strip()

        answers.append(word)

        # Free up memory
        torch.cuda.empty_cache()
        del model_inputs
        del generated_ids
        del decoded
        del encodeds

    return answers


def train_sequence_classifier(
    model_args: TransformerArguments,
    train_args: TrainingArguments,
    data_args: DataArguments,
    lora_args: LoraArguments,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
    max_length: int,
) -> Tuple[PreTrainedModel, List[int]]:
    """
    Trains and evaluates a sequence classification model.

    Args:
        model_args: Configuration for the transformer model.
        train_args: Training configuration.
        data_args: Data configuration.
        lora_args: LoRA configuration for parameter-efficient fine-tuning.
        train_data: Training data DataFrame.
        valid_data: Validation data DataFrame.
        test_data: Test data DataFrame.
        tokenizer: Tokenizer for the model.
        model: The sequence classification model to train.
        device: Device to train on.
        max_length: Maximum sequence length for tokenization.

    Returns:
        A tuple containing the trained model and predicted labels for test data.
    """
    # Prepare datasets
    train_dataset = Dataset.from_dict(
        {
            "text": list(train_data["text"]),
            "label": list(train_data["label"]),
        }
    )

    valid_dataset = Dataset.from_dict(
        {
            "text": list(valid_data["text"]),
            "label": list(valid_data["label"]),
        }
    )

    # Get test texts and labels
    test_texts = test_data["text"]
    test_labels = test_data["label"]

    # Define tokenize function
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

    # Apply tokenization
    train_dataset = train_dataset.map(tokenize, batched=True)
    valid_dataset = valid_dataset.map(tokenize, batched=True)

    # Configure the model
    model.config.pad_token_id = model.config.eos_token_id

    # Add custom tokens if specified
    if model_args.tokenizer_add_tokens:
        tokenizer.add_tokens(model_args.tokenizer_add_tokens)
        model.resize_token_embeddings(len(tokenizer))

    # Data collator for padding batches
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logger.info(f"Model loaded: {model.base_model_prefix}")

    # Apply LoRA if specified
    if model_args.lora:
        modules = find_all_linear_names(model)
        lora_config = LoraConfig(
            lora_alpha=lora_args.lora_alpha,
            r=lora_args.lora_r,
            lora_dropout=lora_args.lora_dropout,
            task_type=model_args.type,
            target_modules=modules if len(modules) > 0 else None,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Prepare model for 4-bit training if quantization is enabled
    if model_args.quant:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

    # Move model to device
    model.to(device)
    logger.info(f"Model device: {model.device}")

    # Calculate class weights for imbalanced data
    train_df = train_dataset.to_pandas()
    pos_weight = len(train_df) / (2 * train_df.label.value_counts()[1])
    neg_weight = len(train_df) / (2 * train_df.label.value_counts()[0])

    logger.info(f"Positive weight: {pos_weight} | Negative weight: {neg_weight}")

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(3, 0.01)

    # Initialize trainer with weighted loss
    trainer = WeightedCELossTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        pos_weights=pos_weight,
        neg_weights=neg_weight,
        callbacks=[early_stopping],
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()
    trainer.save_model()

    # Push to Hub if requested
    if model_args.push_id:
        trainer.push_to_hub(model_args.push_id)

    # Merge LoRA weights with base model if LoRA was used
    if model_args.lora:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        del model
        model = PeftModel.from_pretrained(base_model, train_args.output_dir)
        model = model.merge_and_unload()

    # Make predictions
    predictions = predict_sequence_classification(
        test_texts, model, tokenizer, device, max_length
    )

    return model, predictions


def train_causal_lm(
    model_args: TransformerArguments,
    train_args: TrainingArguments,
    data_args: DataArguments,
    lora_args: LoraArguments,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
    max_length: int,
) -> Tuple[PreTrainedModel, List[str]]:
    """
    Trains and evaluates a causal language model.

    Args:
        model_args: Configuration for the transformer model.
        train_args: Training configuration.
        data_args: Data configuration.
        lora_args: LoRA configuration for parameter-efficient fine-tuning.
        train_data: Training data DataFrame.
        valid_data: Validation data DataFrame.
        test_data: Test data DataFrame.
        tokenizer: Tokenizer for the model.
        model: The causal language model to train.
        device: Device to train on.
        max_length: Maximum sequence length for tokenization.

    Returns:
        A tuple containing the trained model and predicted text for test data.
    """
    # Validate required arguments
    if not model_args.train_prompt:
        raise ValueError("Training prompt is required for causal language modeling")

    if not model_args.test_prompt:
        raise ValueError("Testing prompt is required for causal language modeling")

    if not model_args.delimiter:
        raise ValueError(
            "Delimiter is required for extracting answers from generated text"
        )

    # Configure model
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_reentrant = False

    # Add custom tokens if specified
    if model_args.tokenizer_add_tokens:
        tokenizer.add_tokens(model_args.tokenizer_add_tokens)
        model.resize_token_embeddings(len(tokenizer))

    logger.info(f"Model loaded: {model.base_model_prefix}")

    # Rename columns for consistency
    train_data.columns = ["id", "text", "label"]
    valid_data.columns = ["id", "text", "label"]
    test_data.columns = ["id", "text", "label"]

    test_labels = list(test_data["label"])

    # Format prompts
    def get_prompt(data, prompt_template):
        return prompt_template.format(data["text"], data["label"])

    # Apply prompt templates
    train_data["prompt"] = train_data.apply(
        lambda row: get_prompt(row, model_args.train_prompt), axis=1
    )
    valid_data["prompt"] = valid_data.apply(
        lambda row: get_prompt(row, model_args.train_prompt), axis=1
    )

    # Format test prompts
    test_texts = test_data["text"].apply(
        lambda text: model_args.test_prompt.format(text)
    )

    # Create HF datasets
    train_dataset = Dataset.from_pandas(train_data[["prompt"]], split="train")
    valid_dataset = Dataset.from_pandas(valid_data[["prompt"]], split="valid")

    # Prepare model for 4-bit training if quantization is enabled
    if model_args.quant:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA if specified
    lora_config = None
    if model_args.lora:
        modules = find_all_linear_names(model)
        logger.info(f"LoRA target modules: {modules}")

        if len(modules) == 0:
            model_args.lora = False
            logger.info("No linear modules found, training with full weights")
        else:
            lora_config = LoraConfig(
                lora_alpha=lora_args.lora_alpha,
                r=lora_args.lora_r,
                lora_dropout=lora_args.lora_dropout,
                task_type=model_args.type,
                target_modules=modules,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    logger.info(f"Model device: {model.device}")

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(3, 0.01)

    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        dataset_text_field="prompt",
        peft_config=lora_config if model_args.lora else None,
        max_seq_length=max_length,
        callbacks=[early_stopping],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    trainer.save_model()

    # Push to Hub if requested
    if model_args.push_id:
        trainer.push_to_hub(model_args.push_id)

    # Merge LoRA weights with base model if LoRA was used
    if model_args.lora:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        del model
        model = PeftModel.from_pretrained(base_model, train_args.output_dir)
        model = model.merge_and_unload()

    # Generate predictions
    predictions = predict_causal_lm(
        list(test_texts), model, tokenizer, device, model_args.delimiter
    )

    return model, predictions


def calculate_metrics(
    test_labels: List[int], predictions: List[int]
) -> Dict[str, List[float]]:
    """
    Calculates evaluation metrics for classification predictions.

    Args:
        test_labels: Ground truth labels.
        predictions: Predicted labels.

    Returns:
        Dictionary with precision, recall, and F1 scores.
    """
    return {
        "precision": [precision_score(test_labels, predictions, average="macro")],
        "recall": [recall_score(test_labels, predictions, average="macro")],
        "f1_score": [f1_score(test_labels, predictions, average="macro")],
    }


def save_results(
    predictions: Union[List[int], List[str]],
    data_args: DataArguments,
    model_args: TransformerArguments,
    source: str,
    model_name: str,
    test_labels: Optional[List[int]] = None,
) -> None:
    """
    Saves prediction results and optionally evaluation metrics.

    Args:
        predictions: Model predictions.
        data_args: Data configuration.
        model_args: Model configuration.
        source: Data source identifier.
        model_name: Name of the model.
        test_labels: Ground truth labels (for calculating metrics).
    """
    locale = Path(data_args.train_file).parts[-3]
    # Ensure result directory exists
    result_dir = (
        Path(data_args.result_dir)
        / model_name
        / data_args.task.title()
        / locale
        / source
    )

    # Save metrics and predictions
    metrics_path = result_dir / "metric.csv"

    prediction_path = result_dir / "prediction.csv"
    pd.DataFrame({"prediction": predictions}).to_csv(prediction_path, index=False)
    logger.info(f"Predictions saved to {prediction_path}")

    # Calculate and save metrics if labels are provided
    if test_labels is not None and model_args.type == "SEQ_CLS":
        metrics = calculate_metrics(test_labels, cast(List[int], predictions))
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")


def setup_device() -> torch.device:
    """
    Sets up the appropriate device for training and inference.

    Returns:
        The selected device (CUDA GPU or CPU).
    """
    if "WORLD_SIZE" in os.environ:
        # Distributed training setup
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        logger.info(f"Using CUDA device {local_rank} for distributed training")
    else:
        # Single device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

    return device


def setup_tokenizer(model_args: TransformerArguments) -> PreTrainedTokenizer:
    """
    Sets up the tokenizer with appropriate configuration.

    Args:
        model_args: Model configuration.

    Returns:
        The configured tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Ensure pad token is available
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = model_args.tokenizer_pad_side
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    return tokenizer


def setup_quantization(
    model_args: TransformerArguments,
    quant_args: QuantArguments,
) -> Optional[BitsAndBytesConfig]:
    """
    Sets up quantization configuration if enabled.

    Args:
        model_args: Model configuration.

    Returns:
        BitsAndBytesConfig for quantization or None if quantization is disabled.
    """
    if not model_args.quant:
        return None

    if not torch.cuda.is_available():
        logger.warning("CUDA required for quantization, disabling quantization")
        model_args.quant = False
        return None

    logger.info("Setting up 4-bit quantization")
    return BitsAndBytesConfig(
        load_in_4bit=quant_args.load_in_4bit,
        bnb_4bit_use_double_quant=quant_args.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=quant_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=quant_args.bnb_4bit_compute_dtype,
    )


def main() -> None:
    """
    Main function orchestrating the training and evaluation workflow.

    The function:
    1. Parses command-line arguments or config file
    2. Sets up the environment (device, seed, logging)
    3. Loads and processes data
    4. Configures and loads the model and tokenizer
    5. Trains the model based on the task type
    6. Evaluates the model on test data
    7. Saves predictions and metrics
    """
    # Parse arguments
    parser = HfArgumentParser(
        (
            DataArguments,
            TrainingArguments,
            TransformerArguments,
            QuantArguments,
            LoraArguments,
        )
    )

    # Parse from JSON file or command line
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args, train_args, model_args, quant_args, lora_args = (
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        )
    else:
        data_args, train_args, model_args, quant_args, lora_args = (
            parser.parse_args_into_dataclasses()
        )

    # Authenticate with HuggingFace Hub if token is provided
    if model_args.token:
        login(token=model_args.token, add_to_git_credential=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # Send telemetry
    send_example_telemetry(f"run_{data_args.task}", model_args, data_args)

    # Set random seed for reproducibility
    set_random_seed(train_args.seed)

    # Setup device (CUDA/CPU)
    device = setup_device()

    # Log training setup details
    logger.info(f"Model name/path: {model_args.model_name_or_path}")
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, "
        f"n_gpu: {train_args.n_gpu}, distributed training: "
        f"{train_args.parallel_mode.value == 'distributed'}, 16-bits training: {train_args.fp16}"
    )

    # Determine maximum sequence length
    max_length = max(256, data_args.max_seq_length)

    # Load data
    train_data = get_data(data_args.train_file)
    valid_data = get_data(data_args.valid_file)
    test_data = None

    if data_args.test_file:
        test_data = get_data(data_args.test_file)
    else:
        logger.warning("No test file provided, skipping evaluation")
        return

    # Extract source and model identifiers for file naming
    source = data_args.train_file.split("/")[-1].split(".")[0].strip()
    model_name = model_args.model_name_or_path.split("/")[-1].strip()

    # Setup tokenizer
    tokenizer = setup_tokenizer(model_args)

    # Setup quantization
    bnb_config = setup_quantization(model_args, quant_args)

    # Configure model loading arguments
    model_kwargs = {
        "torch_dtype": model_args.torch_dtype,
        "attn_implementation": model_args.attn_implementation,
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    # Train different model types based on the task
    if model_args.type == "SEQ_CLS":
        # Load sequence classification model
        model_kwargs["num_labels"] = model_args.num_labels

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )

        # Train and evaluate
        model, predictions = train_sequence_classifier(
            model_args=model_args,
            train_args=train_args,
            data_args=data_args,
            lora_args=lora_args,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
        )

        # Save results
        save_results(
            predictions=predictions,
            data_args=data_args,
            model_args=model_args,
            source=source,
            model_name=model_name,
            test_labels=test_data["label"],
        )

    elif model_args.type == "CAUSAL_LM":
        # Load causal language model
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )

        # Train and evaluate
        model, predictions = train_causal_lm(
            model_args=model_args,
            train_args=train_args,
            data_args=data_args,
            lora_args=lora_args,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
        )

        # Save results
        save_results(
            predictions=predictions,
            data_args=data_args,
            model_args=model_args,
            source=source,
            model_name=model_name,
        )

    else:
        raise ValueError(
            f"Unsupported task type: {model_args.type}. "
            f"Supported types: 'SEQ_CLS' and 'CAUSAL_LM'"
        )


if __name__ == "__main__":
    main()

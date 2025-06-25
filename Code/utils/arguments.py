from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch

MODEL_TYPES = ["lr", "svm", "nn", "transformer"]
TASK_TYPES = ["SEQ_CLS", "CAUSAL_LM"]
EMBED_TYPES = ["tfidf", "w2v", "glove", "transformer"]


@dataclass
class DataArguments:

    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or json file)."},
    )
    valid_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input validation data file to evaluate on (a csv or json file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv or json file)."
        },
    )

    result_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to store predictions and metrics obtained for the test_file."
        },
    )
    task: Optional[str] = field(
        default="sentiment",
        metadata={
            "help": "Choose the task to perform. Options: `Sarcasm Detection` or `Sentiment Classification`."
        },
    )
    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )

    def __post_init__(self):

        if (
            # self.dataset_name is None and
            self.train_file is None
            and self.valid_file is None
            and self.test_file is None
        ):
            raise ValueError("Need a training and validation or testing file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "jsonl"]:
                    raise ValueError("`train_file` should be a csv or a json file.")
            if self.valid_file is not None:
                extension = self.valid_file.split(".")[-1]
                if extension not in ["csv", "json", "jsonl"]:
                    raise ValueError(
                        "`validation_file` should be a csv or a json file."
                    )
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                if extension not in ["csv", "json", "jsonl"]:
                    raise ValueError("`test_file` should be a csv or a json file.")


@dataclass
class ModelTypeArguments:

    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose the following model types for experiments: "
            + ", ".join(MODEL_TYPES)
        },
    )


@dataclass
class EmbedArguments:

    embed_type: Optional[str] = field(
        default=None, metadata={"help": "Type of embedding to choose from"}
    )

    embed_path: Optional[str] = field(
        default=None, metadata={"help": "Path to embeddings file/folder"}
    )


@dataclass
class LogisticArguments:
    param_search: Optional[str] = field(
        default=None,
        metadata={
            "help": "It is an option to perform hyperparameter optimisation for the model types: 'LOGISTIC' and 'SVM'. Options: ['random', 'grid']"
        },
    )
    penalty: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Penalty norm for Logistic Regression. Options: ['l1', 'l2', 'elasticnet', None]."
        },
    )
    tol: Optional[List[float]] = field(
        default=None, metadata={"help": "Tolerance for stopping criteria."}
    )

    C: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization."
        },
    )

    max_iter: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Maximum number of iterations taken for the solvers to converge."
        },
    )

    n_jobs: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Number of CPU cores used when parallelizing over classes if multi_class='ovr'”."
        },
    )

    l1_ratio: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2."
        },
    )


@dataclass
class SVMArguments:
    C: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty."
        },
    )

    tol: Optional[List[float]] = field(
        default=None, metadata={"help": "Tolerance for stopping criteria."}
    )


@dataclass
class NetworkArguments:
    pass


@dataclass
class TransformerArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a HuggingFace model from scratch."
            )
        },
    )

    type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Task to choose the head over the transformer model"
            + ", ".join(TASK_TYPES)
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    low_cpu_mem_usage: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    lora: Optional[bool] = field(
        default=False, metadata={"help": "Set this flag to train using LoRA weights."}
    )

    quant: Optional[bool] = field(
        default=False, metadata={"help": "Set this flag to enable quantization."}
    )

    tokenizer_pad_side: Optional[str] = field(
        default="right",
        metadata={"help": "Tokenizer pads the sequence on the specified side."},
    )

    tokenizer_add_tokens: Optional[List[str]] = field(
        default=None, metadata={"help": "Add tokens to the tokenizer"}
    )

    num_labels: Optional[int] = field(
        default=None, metadata={"help": "Number of labels to be classified"}
    )

    train_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": "Training prompt for containing instruction/phrase for causal language modelling."
        },
    )

    test_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": "Testing prompt for containing instruction/phrase for causal language modelling."
        },
    )

    delimiter: Optional[str] = field(
        default=None,
        metadata={"help": "Delimiter to clean outputs from causal language models."},
    )

    shots: Optional[int] = field(
        default=0,
        metadata={
            "help": "Number of examples to be provided in the prompt. `0` for `zero shot prompting`, and so on."
        },
    )
    push_id: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace Repo ID to push the fine-tuned model to."},
    )

    def __post_init__(self):
        if self.type == "SEQ_CLS":
            if self.num_labels is None:
                raise ValueError(
                    "Number of labels required for --type=`SEQ_CLS`. Provide them as --num_labels [VALUE]."
                )

        if self.type == "CAUSAL_LM":
            if self.train_prompt is None or self.test_prompt is None:
                raise ValueError(
                    "Prompt required for --type=`CAUSAL_LM`. Provide them as --prompt [VALUE]."
                )
            if self.delimiter is None:
                raise ValueError(
                    "Delimiter required for --type=`CAUSAL_LM`. Provide them as --delimiter [VALUE]."
                )


@dataclass
class LoraArguments:
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: Optional[int] = field(default=8, metadata={"help": "The r value for lora"})

    lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "The alpha value for lora"}
    )

    stop_after_n_steps: Optional[int] = field(
        default=10000, metadata={"help": "Stop training after n steps"}
    )


@dataclass
class QuantArguments:
    load_in_4bit: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes."
        },
    )

    bnb_4bit_quant_type: Optional[str] = field(
        default="fp4",
        metadata={
            "help": "This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by `fp4` or `nf4`."
        },
    )

    bnb_4bit_use_double_quant: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This flag is used for nested quantization where the quantization constants from the first quantization are quantized again."
        },
    )

    bnb_4bit_compute_dtype: Optional[Union[torch.dtype, str]] = field(
        default=torch.float32,
        metadata={
            "help": "This sets the computational type which might be different than the input type. For example, inputs might be fp32, but computation can be set to bf16 for speedups."
        },
    )

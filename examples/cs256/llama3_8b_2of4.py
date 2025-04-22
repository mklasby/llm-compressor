import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.modifiers.obcq import SparseGPTModifier
from llmcompressor.modifiers.pruning import ConstantPruningModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot

# Configuration
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Apply compression to a model")
    parser.add_argument(
        "--fp8", action="store_true", help="Enable FP8 compression"
    )
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument("--sparsity", type=float, default=0.0)
    return parser.parse_args()


def preprocess(example):
    """Preprocess dataset examples."""
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
    }


def tokenize(sample):
    """Tokenize dataset examples."""
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


def get_recipe(fp8_enabled, mask_structure, sparsity, block_size):
    """Generate the compression recipe and save directory based on the FP8 flag."""
    base_recipe = [
        SparseGPTModifier(
            sparsity=sparsity,
            mask_structure=mask_structure,
            sequential_update=True,
            targets=[r"re:model.layers.\d*$"],
            block_size=256,
        )
    ]
    save_dir = (
        MODEL_ID.split("/")[1]
        + f"sparsity-{sparsity:.2f}_{mask_structure.replace(':', 'of')}-sparse_block-size_256"
    )

    if fp8_enabled:
        base_recipe.extend(
            [
                QuantizationModifier(
                    targets=["Linear"],
                    ignore=["lm_head"],
                    scheme="FP8_DYNAMIC",
                ),
                ConstantPruningModifier(
                    targets=[
                        r"re:.*q_proj.weight",
                        r"re:.*k_proj.weight",
                        r"re:.*v_proj.weight",
                        r"re:.*o_proj.weight",
                        r"re:.*gate_proj.weight",
                        r"re:.*up_proj.weight",
                        r"re:.*down_proj.weight",
                    ],
                    start=0,
                ),
            ]
        )
        # save_dir = MODEL_ID.split("/")[1] + "2of4-W8A8-FP8-Dynamic-Per-Token"
        save_dir = (
            MODEL_ID.split("/")[1]
            + f"{mask_structure.replace(':', 'of')}-W8A8-FP8-Dynamic-Per-Token"
        )

    return base_recipe, save_dir


# Parse arguments
args = parse_args()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load and preprocess dataset
ds = (
    load_dataset(DATASET_ID, split=DATASET_SPLIT)
    .shuffle(seed=42)
    .select(range(NUM_CALIBRATION_SAMPLES))
)
ds = ds.map(preprocess)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# Get compression recipe and save directory
mask_structure = f"{args.prunen}:{args.prunem}"
if mask_structure != "0:0":
    sparsity = args.prunen / args.prunem
else:
    sparsity = args.sparsity
block_size = 256
if args.prunem > block_size:
    block_size = args.prunem
recipe, save_dir = get_recipe(args.fp8, mask_structure, sparsity, block_size)

# Apply compression
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Validate the compressed model
print("\n========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    "cuda"
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n")

# Save compressed model and tokenizer
model.save_pretrained(
    save_dir, save_compressed=args.fp8, disable_sparse_compression=True
)
tokenizer.save_pretrained(save_dir)

import torch


def get_sparsity(w):
    n_zeros = (w == 0).sum()
    return n_zeros / w.numel()


for n, m in model.named_modules():
    if "embedding" in n or "lm_head" in n:
        continue
    if isinstance(m, torch.nn.Linear):
        sparsity = get_sparsity(m.weight)
        print(n, sparsity)
        break

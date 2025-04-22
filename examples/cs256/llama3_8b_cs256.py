import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


from llmcompressor.modifiers.obcq import SparseGPTTiledModifier
from llmcompressor.transformers import oneshot

# Configuration
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
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
    parser.add_argument("--tile-width", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--sparsity", type=float, default=0.5)
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


def get_recipe(fp8_enabled, sparsity, tile_width, block_size=256, group_size=8):
    """Generate the compression recipe and save directory based on the FP8 flag."""
    base_recipe = [
        SparseGPTTiledModifier(
            sparsity=sparsity,
            sequential_update=True,
            targets=[r"re:model.layers.\d*$"],
            block_size=block_size,
            preserve_sparsity_mask=False,
            group_size=group_size,
        )
    ]
    save_dir = (
        MODEL_ID.split("/")[1]
        + f"_cs{tile_width}_block-size-{block_size}_group-{group_size}_sparsity-{sparsity:.2f}"
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
recipe, save_dir = get_recipe(
    args.fp8, args.sparsity, args.tile_width, 256, args.group_size
)


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
    save_dir, save_compressed=False, disable_sparse_compression=True
)
tokenizer.save_pretrained(save_dir)

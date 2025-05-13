"""
python your_script_name.py \
    --model_stub "neuralmagic/Llama-2-7b-ultrachat200k" \
    --dataset "ultrachat-200k" \
    --output_dir "./output_llama7b_2of4_w4a16_oneshot" \
    --recipe "recipes/llama2_2of4_w4a16_channel_oneshot.yaml" \
    --num_calibration_samples 512 \
    --max_seq_length 512
    
    
python sparse_drafter.py \
    --model_stub "/home/mike/self-distilled-sparse-drafters/artifacts/models/meta-llama/Llama-3.2-1B-Instruct/sparsegpt/nm_24_sparse_ft" \
    --dataset "ultrachat-200k" \
    --output_dir "/home/mike/self-distilled-sparse-drafters/artifacts/models/meta-llama/Llama-3.2-1B-Instruct/sparsegpt/nm_24_sparse_ft/2of4_w4a16_oneshot" \
    --recipe "2of4_w4a16_group-128_recipe.yaml" \
    --num_calibration_samples 512 \
    --max_seq_length 512
"""
import argparse
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot


def main(args):
    # load the model in as bfloat16 to save on memory and compute
    logger.info(f"Loading model: {args.model_stub}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_stub, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    # tokenizer = AutoTokenizer.from_pretrained(args.model_stub)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # set dataset config parameters
    # The dataset will only be used for calibration
    splits = {"calibration": args.calibration_split}

    logger.info(f"Using dataset: {args.dataset} for calibration")
    logger.info(f"Calibration split: {args.calibration_split}")
    logger.info(f"Recipe for sparsification and quantization: {args.recipe}")
    logger.info(f"Output directory: {args.output_dir}")

    oneshot_kwargs = dict(
        dataset=args.dataset,
        recipe=args.recipe,
        num_calibration_samples=args.num_calibration_samples,
        preprocessing_num_workers=args.preprocessing_num_workers,
        splits=splits,
        max_seq_length=args.max_seq_length, # Added for oneshot data processing
        trust_remote_code_model=True,
        processor=tokenizer,
    )

    # This will run the targeted stage of the recipe
    # oneshot sparsification -> oneshot quantization

    # Models are automatically saved in
    # ./<output_dir>/ + (sparsity/quantization)_stage

    # Oneshot sparsification
    logger.info("Starting one-shot sparsification...")
    sparsified_model = oneshot(
        model=model,
        **oneshot_kwargs,
        stage="sparsity_stage", # Assuming recipe has a 'sparsity_stage'
        output_dir=args.output_dir,
    )
    logger.info(f"One-shot sparsification complete. Model saved in {args.output_dir}/sparsity_stage")

    # Oneshot quantization
    logger.info("Starting one-shot quantization...")
    quantized_model = oneshot(
        model=sparsified_model, # Use the sparsified model as input
        **oneshot_kwargs,
        stage="quantization_stage", # Assuming recipe has a 'quantization_stage'
        output_dir=args.output_dir, # Output for this stage will also go under args.output_dir
                                     # llmcompressor typically creates subdirs like 'quantization_stage'
    )

    final_quantized_path = f"{args.output_dir}/quantization_stage"
    logger.info(f"One-shot quantization complete. Model saved in {args.output_dir}/quantization_stage")
    quantized_model.save_pretrained(
        final_quantized_path, skip_sparsity_compression_stats=False
    )
    tokenizer.save_pretrained(final_quantized_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one-shot sparsification and quantization on a language model.")

    # Required arguments
    parser.add_argument("--model_stub", type=str, required=True, help="Model stub for Hugging Face model hub (e.g., 'neuralmagic/Llama-2-7b-ultrachat200k')")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name for calibration (e.g., 'ultrachat-200k')")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output quantized model")

    # Arguments with default values (for one-shot sparsification/quantization)
    parser.add_argument("--recipe", type=str, default="2of4_w4a16_group-128_recipe.yaml", help="Recipe file for sparsification and quantization")
    parser.add_argument("--num_calibration_samples", type=int, default=512, help="Number of samples for calibration")
    parser.add_argument("--preprocessing_num_workers", type=int, default=64, help="Number of workers for dataset preprocessing")
    parser.add_argument("--calibration_split", type=str, default="train_gen[:5%]", help="Dataset split to use for calibration (e.g., 'train_gen[:5%%]')")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for processing calibration data")

    parsed_args = parser.parse_args()
    main(parsed_args)

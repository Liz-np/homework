"""
Analysis script for Part 1: Identifying artifacts and hard slices.
Run this in Colab after training your baseline model.
"""

import json
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from collections import defaultdict


def has_negation(example):
    """Check if example contains negation words."""
    text = (example.get("premise", "") + " " + example.get("hypothesis", "")).lower()
    negation_words = r"\b(not|no|never|nobody|nothing|none|neither|nowhere|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b"
    return bool(re.search(negation_words, text))


def has_long_premise(example, min_length=20):
    """Check if premise is long (by word count)."""
    premise = example.get("premise", "")
    return len(premise.split()) >= min_length


def compute_lexical_overlap(example):
    """Compute word overlap between premise and hypothesis."""
    premise_words = set(example.get("premise", "").lower().split())
    hypothesis_words = set(example.get("hypothesis", "").lower().split())
    if len(premise_words) == 0 or len(hypothesis_words) == 0:
        return 0.0
    overlap = len(premise_words & hypothesis_words)
    union = len(premise_words | hypothesis_words)
    return overlap / union if union > 0 else 0.0


def evaluate_on_slice(model, tokenizer, dataset_slice, device="cuda", batch_size=32):
    """Evaluate model on a dataset slice."""
    model.eval()
    predictions = []
    gold_labels = []
    
    # Convert dataset to list for easier batch processing
    dataset_list = [dataset_slice[i] for i in range(len(dataset_slice))]
    
    # Process in batches
    for i in tqdm(range(0, len(dataset_list), batch_size)):
        batch = dataset_list[i:i+batch_size]
        
        # Extract data from batch
        premises = [ex["premise"] for ex in batch]
        hypotheses = [ex["hypothesis"] for ex in batch]
        labels = [ex["label"] for ex in batch]
        
        # Tokenize
        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
        
        predictions.extend(batch_preds)
        gold_labels.extend(labels)
    
    # Compute accuracy
    accuracy = np.mean(np.array(predictions) == np.array(gold_labels))
    
    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "gold_labels": gold_labels,
        "num_examples": len(dataset_slice)
    }


def collect_error_examples(dataset_slice, predictions, gold_labels, num_examples=10):
    """Collect examples where model made errors."""
    errors = []
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    # Convert dataset to list for iteration
    dataset_list = [dataset_slice[i] for i in range(len(dataset_slice))]
    
    for i, (ex, pred, gold) in enumerate(zip(dataset_list, predictions, gold_labels)):
        if pred != gold and len(errors) < num_examples:
            errors.append({
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "gold_label": label_map[gold],
                "predicted_label": label_map[pred],
                "index": i
            })
    
    return errors


def create_slices(dataset):
    """Create different slices of the dataset."""
    slices = {}
    
    # Negation slice
    negation_slice = dataset.filter(has_negation)
    slices["negation"] = negation_slice
    print(f"[OK] Negation slice: {len(negation_slice)} examples")
    
    # Non-negation slice
    non_negation_slice = dataset.filter(lambda x: not has_negation(x))
    slices["non_negation"] = non_negation_slice
    print(f"[OK] Non-negation slice: {len(non_negation_slice)} examples")
    
    # Long premise slice
    long_premise_slice = dataset.filter(has_long_premise)
    slices["long_premise"] = long_premise_slice
    print(f"[OK] Long premise slice: {len(long_premise_slice)} examples")
    
    # Short premise slice
    short_premise_slice = dataset.filter(lambda x: not has_long_premise(x))
    slices["short_premise"] = short_premise_slice
    print(f"[OK] Short premise slice: {len(short_premise_slice)} examples")
    
    # High lexical overlap
    dataset_with_overlap = dataset.map(
        lambda x: {"lexical_overlap": compute_lexical_overlap(x)}
    )
    high_overlap_slice = dataset_with_overlap.filter(lambda x: x["lexical_overlap"] >= 0.5)
    slices["high_overlap"] = high_overlap_slice
    print(f"[OK] High lexical overlap slice: {len(high_overlap_slice)} examples")
    
    # Low lexical overlap
    low_overlap_slice = dataset_with_overlap.filter(lambda x: x["lexical_overlap"] < 0.3)
    slices["low_overlap"] = low_overlap_slice
    print(f"[OK] Low lexical overlap slice: {len(low_overlap_slice)} examples")
    
    return slices


def visualize_results(results_dict, output_path="analysis_results.png"):
    """Create visualizations of slice performance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart comparing slice accuracies
    slice_names = list(results_dict.keys())
    accuracies = [results_dict[name]["accuracy"] for name in slice_names]
    
    axes[0].bar(range(len(slice_names)), accuracies, color=['red' if acc < 0.8 else 'green' for acc in accuracies])
    axes[0].set_xticks(range(len(slice_names)))
    axes[0].set_xticklabels(slice_names, rotation=45, ha='right')
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy by Slice")
    axes[0].axhline(y=0.8, color='orange', linestyle='--', label='80% threshold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Comparison: Negation vs Non-negation
    if "negation" in results_dict and "non_negation" in results_dict:
        comparison_data = {
            "Negation": results_dict["negation"]["accuracy"],
            "Non-negation": results_dict["non_negation"]["accuracy"]
        }
        axes[1].bar(comparison_data.keys(), comparison_data.values(), 
                   color=['red', 'green'], alpha=0.7)
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Negation vs Non-negation Performance")
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for k, v in comparison_data.items():
            axes[1].text(k, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved visualization to {output_path}")
    plt.show()


def main():
    """Main analysis workflow."""
    print("=" * 60)
    print("Part 1: Dataset Artifact Analysis")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading SNLI dataset...")
    dataset = load_dataset("snli")
    val_dataset = dataset["validation"].filter(lambda x: x["label"] != -1)
    print(f"[OK] Validation set: {len(val_dataset)} examples")
    
    # Load baseline model - check multiple locations
    print("\n2. Loading baseline model...")
    
    # Check for model in different locations
    model_path = None
    possible_paths = [
        "./trained_snli_baseline",
        "./results/baseline",
        "../trained_snli_baseline",
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
            # Check if it has model weights
            has_weights = (os.path.exists(os.path.join(path, "model.safetensors")) or
                         os.path.exists(os.path.join(path, "pytorch_model.bin")) or
                         os.path.exists(os.path.join(path, "checkpoint-99500", "model.safetensors")))
            if has_weights:
                model_path = path
                print(f"[OK] Found model with weights at: {path}")
                break
            elif path == "./results/baseline":
                # Results folder has config but may not have weights
                print(f"[WARNING]  Found config at {path} but checking for weights...")
                model_path = path  # Will try to load, may fall back to pre-trained
    
    if not model_path:
        print("[WARNING]  No local model found. Using pre-trained ELECTRA-small for testing.")
        print("   (Results will differ from your trained model)")
        model_path = "google/electra-small-discriminator"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        # Use tokenizer from results if available, otherwise from model path
        tokenizer_path = "./results/baseline" if os.path.exists("./results/baseline/tokenizer.json") else model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"[OK] Model loaded on {device}")
    except Exception as e:
        print(f"[ERROR] Error loading model from {model_path}: {e}")
        print("   Trying pre-trained model as fallback...")
        model_path = "google/electra-small-discriminator"
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"[OK] Loaded pre-trained model on {device}")
    
    # Create slices
    print("\n3. Creating dataset slices...")
    slices = create_slices(val_dataset)
    
    # Evaluate on each slice
    print("\n4. Evaluating model on slices...")
    results = {}
    
    # Evaluate on full validation set first
    print("  Evaluating on full validation set...")
    full_results = evaluate_on_slice(model, tokenizer, val_dataset, device=device)
    results["full_validation"] = full_results
    print(f"    Accuracy: {full_results['accuracy']:.4f}")
    
    # Evaluate on each slice
    for slice_name, slice_data in slices.items():
        print(f"  Evaluating on {slice_name} slice ({len(slice_data)} examples)...")
        slice_results = evaluate_on_slice(model, tokenizer, slice_data, device=device)
        results[slice_name] = slice_results
        print(f"    Accuracy: {slice_results['accuracy']:.4f}")
    
    # Collect error examples
    print("\n5. Collecting error examples...")
    error_examples = {}
    for slice_name in ["negation", "long_premise", "high_overlap"]:
        if slice_name in slices and slice_name in results:
            errors = collect_error_examples(
                slices[slice_name],
                results[slice_name]["predictions"],
                results[slice_name]["gold_labels"],
                num_examples=10
            )
            error_examples[slice_name] = errors
            print(f"  [OK] Collected {len(errors)} error examples from {slice_name} slice")
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    visualize_results(results, output_path="analysis_results.png")
    
    # Save results
    print("\n7. Saving results...")
    output_data = {
        "full_validation_accuracy": results["full_validation"]["accuracy"],
        "slice_results": {
            name: {
                "accuracy": res["accuracy"],
                "num_examples": res["num_examples"]
            }
            for name, res in results.items() if name != "full_validation"
        },
        "error_examples": error_examples
    }
    
    with open("analysis_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("[OK] Saved results to analysis_results.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall Validation Accuracy: {results['full_validation']['accuracy']:.4f}")
    print("\nSlice Performance:")
    for name, res in results.items():
        if name != "full_validation":
            print(f"  {name:20s}: {res['accuracy']:.4f} ({res['num_examples']} examples)")
    
    if "negation" in results and "non_negation" in results:
        gap = results["non_negation"]["accuracy"] - results["negation"]["accuracy"]
        print(f"\n[WARNING]  Performance gap (non-negation - negation): {gap:.4f}")
        if gap > 0.1:
            print("   Significant performance drop on negation examples!")
    
    print("\n✅ Analysis complete!")
    print("   - Check analysis_results.json for detailed results")
    print("   - Check analysis_results.png for visualizations")
    print("   - Error examples are saved in analysis_results.json")


if __name__ == "__main__":
    main()


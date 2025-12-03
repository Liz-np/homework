"""
Enhanced Analysis Script for Part 1: Multiple Approaches
Implements:
1. Contrast Sets (Gardner et al., 2020) - modifying examples slightly
2. Model Ablations (Poliak et al., 2018) - hypothesis-only baseline
3. Statistical Tests (Gardner et al., 2021) - n-gram correlations
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
from collections import defaultdict, Counter
import random


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


# ============================================================================
# APPROACH 1: CONTRAST SETS (Gardner et al., 2020)
# ============================================================================

def create_contrast_examples(example, num_variants=3):
    """
    Create contrast examples by modifying the original example slightly.
    Returns list of modified examples with same gold label.
    """
    contrast_examples = []
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label = example["label"]
    
    # Strategy 1: Add negation to hypothesis
    if "not" not in hypothesis.lower():
        negated_hyp = hypothesis.replace(" is ", " is not ", 1)
        negated_hyp = negated_hyp.replace(" are ", " are not ", 1)
        if negated_hyp != hypothesis:
            contrast_examples.append({
                "premise": premise,
                "hypothesis": negated_hyp,
                "label": label,  # Keep original label (may need manual correction)
                "contrast_type": "negation_added"
            })
    
    # Strategy 2: Replace key words with synonyms/antonyms
    replacements = [
        ("happy", "sad"),
        ("big", "small"),
        ("hot", "cold"),
        ("good", "bad"),
        ("many", "few"),
    ]
    for old_word, new_word in replacements:
        if old_word in hypothesis.lower():
            modified_hyp = hypothesis.replace(old_word, new_word, 1)
            contrast_examples.append({
                "premise": premise,
                "hypothesis": modified_hyp,
                "label": label,  # May need correction
                "contrast_type": f"word_replacement_{old_word}_{new_word}"
            })
            if len(contrast_examples) >= num_variants:
                break
    
    # Strategy 3: Add small phrase
    if len(contrast_examples) < num_variants:
        phrases = ["in the morning", "at the store", "with friends"]
        for phrase in phrases:
            if phrase not in hypothesis.lower():
                modified_hyp = hypothesis + " " + phrase
                contrast_examples.append({
                    "premise": premise,
                    "hypothesis": modified_hyp,
                    "label": label,
                    "contrast_type": "phrase_added"
                })
                break
    
    return contrast_examples[:num_variants]


def evaluate_contrast_sets(model, tokenizer, original_dataset, device="cuda", num_examples=50):
    """
    Create and evaluate contrast sets.
    Returns accuracy on original vs contrast examples.
    """
    print("\n[APPROACH 1] Creating and evaluating contrast sets...")
    
    # Sample examples to create contrasts for
    sample_indices = random.sample(range(len(original_dataset)), min(num_examples, len(original_dataset)))
    sample_dataset = original_dataset.select(sample_indices)
    
    contrast_examples = []
    original_examples = []
    
    for ex in sample_dataset:
        original_examples.append(ex)
        contrasts = create_contrast_examples(ex, num_variants=2)
        contrast_examples.extend(contrasts)
    
    print(f"  Created {len(contrast_examples)} contrast examples from {len(original_examples)} originals")
    
    # Evaluate on originals
    orig_acc = evaluate_slice_simple(model, tokenizer, original_examples, device)
    
    # Evaluate on contrasts
    contrast_acc = evaluate_slice_simple(model, tokenizer, contrast_examples, device)
    
    return {
        "original_accuracy": orig_acc,
        "contrast_accuracy": contrast_acc,
        "performance_drop": orig_acc - contrast_acc,
        "num_original": len(original_examples),
        "num_contrast": len(contrast_examples)
    }


# ============================================================================
# APPROACH 2: MODEL ABLATIONS - Hypothesis-Only Baseline (Poliak et al., 2018)
# ============================================================================

def evaluate_hypothesis_only(model, tokenizer, dataset, device="cuda", batch_size=32):
    """
    Evaluate model using only hypothesis (ignoring premise).
    This tests if model relies on hypothesis-only heuristics.
    """
    print("\n[APPROACH 2] Evaluating hypothesis-only baseline...")
    
    model.eval()
    predictions = []
    gold_labels = []
    
    dataset_list = [dataset[i] for i in range(len(dataset))]
    
    for i in tqdm(range(0, len(dataset_list), batch_size), desc="Hypothesis-only"):
        batch = dataset_list[i:i+batch_size]
        
        # Use empty premise or just hypothesis
        premises = [""] * len(batch)  # Empty premise
        hypotheses = [ex["hypothesis"] for ex in batch]
        labels = [ex["label"] for ex in batch]
        
        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        
        predictions.extend(batch_preds)
        gold_labels.extend(labels)
    
    accuracy = np.mean(np.array(predictions) == np.array(gold_labels))
    
    return {
        "hypothesis_only_accuracy": accuracy,
        "num_examples": len(dataset)
    }


# ============================================================================
# APPROACH 3: STATISTICAL TESTS - N-gram Correlations (Gardner et al., 2021)
# ============================================================================

def find_ngram_correlations(dataset, predictions, gold_labels, n=2, top_k=10):
    """
    Find spurious n-gram correlations with labels.
    Returns n-grams that are highly correlated with specific labels.
    """
    print(f"\n[APPROACH 3] Finding {n}-gram correlations with labels...")
    
    # Collect n-grams from hypotheses for each label
    label_ngrams = {0: Counter(), 1: Counter(), 2: Counter()}  # entailment, neutral, contradiction
    
    dataset_list = [dataset[i] for i in range(len(dataset))]
    
    for ex, pred, gold in zip(dataset_list, predictions, gold_labels):
        hypothesis = ex["hypothesis"].lower()
        words = hypothesis.split()
        
        # Extract n-grams
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            label_ngrams[gold][ngram] += 1
    
    # Find n-grams that are highly correlated with one label
    correlations = []
    
    for label in [0, 1, 2]:
        label_name = ["entailment", "neutral", "contradiction"][label]
        
        # Get total counts for this n-gram across all labels
        all_ngrams = set()
        for l in [0, 1, 2]:
            all_ngrams.update(label_ngrams[l].keys())
        
        for ngram in all_ngrams:
            count_in_label = label_ngrams[label][ngram]
            total_count = sum(label_ngrams[l][ngram] for l in [0, 1, 2])
            
            if total_count >= 5:  # Minimum frequency threshold
                correlation = count_in_label / total_count if total_count > 0 else 0
                if correlation > 0.7:  # High correlation threshold
                    correlations.append({
                        "ngram": ngram,
                        "label": label_name,
                        "correlation": correlation,
                        "count_in_label": count_in_label,
                        "total_count": total_count
                    })
    
    # Sort by correlation
    correlations.sort(key=lambda x: x["correlation"], reverse=True)
    
    return correlations[:top_k]


def evaluate_slice_simple(model, tokenizer, examples_list, device="cuda", batch_size=32):
    """Simple evaluation function for list of examples."""
    model.eval()
    predictions = []
    gold_labels = []
    
    for i in range(0, len(examples_list), batch_size):
        batch = examples_list[i:i+batch_size]
        premises = [ex["premise"] for ex in batch]
        hypotheses = [ex["hypothesis"] for ex in batch]
        labels = [ex["label"] for ex in batch]
        
        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        
        predictions.extend(batch_preds)
        gold_labels.extend(labels)
    
    return np.mean(np.array(predictions) == np.array(gold_labels))


def evaluate_on_slice(model, tokenizer, dataset_slice, device="cuda", batch_size=32):
    """Evaluate model on a dataset slice."""
    model.eval()
    predictions = []
    gold_labels = []
    
    dataset_list = [dataset_slice[i] for i in range(len(dataset_slice))]
    
    for i in tqdm(range(0, len(dataset_list), batch_size), desc="Evaluating"):
        batch = dataset_list[i:i+batch_size]
        premises = [ex["premise"] for ex in batch]
        hypotheses = [ex["hypothesis"] for ex in batch]
        labels = [ex["label"] for ex in batch]
        
        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        
        predictions.extend(batch_preds)
        gold_labels.extend(labels)
    
    accuracy = np.mean(np.array(predictions) == np.array(gold_labels))
    
    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "gold_labels": gold_labels,
        "num_examples": len(dataset_slice)
    }


def create_slices(dataset):
    """Create different slices of the dataset."""
    slices = {}
    
    negation_slice = dataset.filter(has_negation)
    slices["negation"] = negation_slice
    print(f"[OK] Negation slice: {len(negation_slice)} examples")
    
    non_negation_slice = dataset.filter(lambda x: not has_negation(x))
    slices["non_negation"] = non_negation_slice
    print(f"[OK] Non-negation slice: {len(non_negation_slice)} examples")
    
    long_premise_slice = dataset.filter(has_long_premise)
    slices["long_premise"] = long_premise_slice
    print(f"[OK] Long premise slice: {len(long_premise_slice)} examples")
    
    dataset_with_overlap = dataset.map(lambda x: {"lexical_overlap": compute_lexical_overlap(x)})
    high_overlap_slice = dataset_with_overlap.filter(lambda x: x["lexical_overlap"] >= 0.5)
    slices["high_overlap"] = high_overlap_slice
    print(f"[OK] High lexical overlap slice: {len(high_overlap_slice)} examples")
    
    return slices


def main():
    """Main analysis workflow with multiple approaches."""
    print("=" * 60)
    print("Part 1: Enhanced Dataset Artifact Analysis")
    print("Implementing: Contrast Sets, Model Ablations, N-gram Correlations")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading SNLI dataset...")
    dataset = load_dataset("snli")
    val_dataset = dataset["validation"].filter(lambda x: x["label"] != -1)
    print(f"[OK] Validation set: {len(val_dataset)} examples")
    
    # Load model
    print("\n2. Loading baseline model...")
    model_path = None
    possible_paths = ["./trained_snli_baseline", "./results/baseline"]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
            has_weights = (os.path.exists(os.path.join(path, "model.safetensors")) or
                         os.path.exists(os.path.join(path, "pytorch_model.bin")))
            if has_weights:
                model_path = path
                print(f"[OK] Found model with weights at: {path}")
                break
    
    if not model_path:
        print("[WARNING] No local model found. Using pre-trained ELECTRA-small for testing.")
        model_path = "google/electra-small-discriminator"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer_path = "./results/baseline" if os.path.exists("./results/baseline/tokenizer.json") else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"[OK] Model loaded on {device}")
    
    # Basic slice evaluation
    print("\n3. Basic slice evaluation...")
    slices = create_slices(val_dataset)
    results = {}
    
    print("  Evaluating on full validation set...")
    full_results = evaluate_on_slice(model, tokenizer, val_dataset, device=device)
    results["full_validation"] = full_results
    print(f"    Accuracy: {full_results['accuracy']:.4f}")
    
    for slice_name, slice_data in slices.items():
        if len(slice_data) == 0:
            continue
        print(f"  Evaluating on {slice_name} slice...")
        slice_results = evaluate_on_slice(model, tokenizer, slice_data, device=device)
        results[slice_name] = slice_results
        print(f"    Accuracy: {slice_results['accuracy']:.4f}")
    
    # APPROACH 1: Contrast Sets
    contrast_results = evaluate_contrast_sets(model, tokenizer, val_dataset, device=device, num_examples=30)
    results["contrast_sets"] = contrast_results
    print(f"\n  Contrast Sets Results:")
    print(f"    Original accuracy: {contrast_results['original_accuracy']:.4f}")
    print(f"    Contrast accuracy: {contrast_results['contrast_accuracy']:.4f}")
    print(f"    Performance drop: {contrast_results['performance_drop']:.4f}")
    
    # APPROACH 2: Hypothesis-Only Baseline
    hypo_results = evaluate_hypothesis_only(model, tokenizer, val_dataset, device=device)
    results["hypothesis_only"] = hypo_results
    print(f"\n  Hypothesis-Only Results:")
    print(f"    Accuracy: {hypo_results['hypothesis_only_accuracy']:.4f}")
    print(f"    (If high, model may rely on hypothesis-only heuristics)")
    
    # APPROACH 3: N-gram Correlations
    ngram_correlations = find_ngram_correlations(
        val_dataset,
        full_results["predictions"],
        full_results["gold_labels"],
        n=2,
        top_k=15
    )
    results["ngram_correlations"] = ngram_correlations
    print(f"\n  Top N-gram Correlations:")
    for corr in ngram_correlations[:5]:
        print(f"    '{corr['ngram']}' -> {corr['label']} (correlation: {corr['correlation']:.3f}, count: {corr['count_in_label']}/{corr['total_count']})")
    
    # Save results
    print("\n4. Saving results...")
    output_data = {
        "full_validation_accuracy": results["full_validation"]["accuracy"],
        "slice_results": {
            name: {"accuracy": res["accuracy"], "num_examples": res["num_examples"]}
            for name, res in results.items() if name not in ["full_validation", "contrast_sets", "hypothesis_only", "ngram_correlations"]
        },
        "contrast_sets": contrast_results,
        "hypothesis_only": hypo_results,
        "ngram_correlations": ngram_correlations
    }
    
    with open("analysis_results_enhanced.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("[OK] Saved results to analysis_results_enhanced.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy: {results['full_validation']['accuracy']:.4f}")
    print(f"Hypothesis-Only Accuracy: {hypo_results['hypothesis_only_accuracy']:.4f}")
    print(f"Contrast Sets Drop: {contrast_results['performance_drop']:.4f}")
    print(f"Found {len(ngram_correlations)} spurious n-gram correlations")
    print("\n[OK] Enhanced analysis complete!")


if __name__ == "__main__":
    main()


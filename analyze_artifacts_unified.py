"""
Unified Analysis Script for Part 1 and Part 2
Automatically detects and analyzes baseline and/or contrast-augmented models.
Combines functionality from:
- analyze_artifacts.py (basic slice analysis)
- analyze_artifacts_enhanced.py (contrast sets, hypothesis-only, n-grams)
- analyze_artifacts_comprehensive.py (hypothesis length, negation misclassification, logits)
"""

import json
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict, Counter
import random


# ============================================================================
# Helper Functions
# ============================================================================

def has_negation(example):
    """Check if example contains negation words (premise or hypothesis)."""
    text = (example.get("premise", "") + " " + example.get("hypothesis", "")).lower()
    negation_words = r"\b(not|no|never|nobody|nothing|none|neither|nowhere|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b"
    return bool(re.search(negation_words, text))


def has_negation_in_hypothesis(example):
    """Check if hypothesis contains negation words."""
    hypothesis = example.get("hypothesis", "").lower()
    negation_words = r"\b(not|no|never|nobody|nothing|none|neither|nowhere|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b"
    return bool(re.search(negation_words, hypothesis))


def has_long_premise(example, min_length=20):
    """Check if premise is long (by word count)."""
    premise = example.get("premise", "")
    return len(premise.split()) >= min_length


def get_hypothesis_length(example):
    """Get hypothesis length in tokens (words)."""
    hypothesis = example.get("hypothesis", "")
    return len(hypothesis.split())


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
# Evaluation Functions
# ============================================================================

def evaluate_on_slice(model, tokenizer, dataset_slice, device="cuda", batch_size=32, return_logits=False):
    """Evaluate model on a dataset slice."""
    model.eval()
    predictions = []
    gold_labels = []
    all_logits = [] if return_logits else None
    all_probs = [] if return_logits else None
    
    dataset_list = [dataset_slice[i] for i in range(len(dataset_slice))]
    
    for i in tqdm(range(0, len(dataset_list), batch_size), desc="Evaluating slice"):
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
            logits = outputs.logits
            if return_logits:
                probs = F.softmax(logits, dim=-1)
                all_logits.extend(logits.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
        
        predictions.extend(batch_preds)
        gold_labels.extend(labels)
    
    accuracy = np.mean(np.array(predictions) == np.array(gold_labels))
    
    result = {
        "accuracy": accuracy,
        "predictions": predictions,
        "gold_labels": gold_labels,
        "num_examples": len(dataset_slice)
    }
    
    if return_logits:
        result["logits"] = all_logits
        result["probabilities"] = all_probs
    
    return result


def evaluate_with_logits(model, tokenizer, examples_list, device="cuda", batch_size=32):
    """Evaluate model and return predictions, gold labels, and logits/softmax probabilities."""
    model.eval()
    predictions = []
    gold_labels = []
    all_logits = []
    all_probs = []
    
    for i in tqdm(range(0, len(examples_list), batch_size), desc="Evaluating with logits"):
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
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
        
        predictions.extend(batch_preds)
        gold_labels.extend(labels)
        all_logits.extend(logits.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return {
        "predictions": predictions,
        "gold_labels": gold_labels,
        "logits": all_logits,
        "probabilities": all_probs
    }


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


def collect_error_examples(dataset_slice, predictions, gold_labels, num_examples=10):
    """Collect examples where model made errors."""
    errors = []
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
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


# ============================================================================
# Slice Creation
# ============================================================================

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


# ============================================================================
# Visualization
# ============================================================================

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
    plt.close()


# ============================================================================
# APPROACH 1: CONTRAST SETS (Gardner et al., 2020)
# ============================================================================

def create_contrast_examples(example, num_variants=3):
    """Create contrast examples by modifying the original example slightly."""
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
                "label": label,
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
                "label": label,
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


def create_contrast_pairs(example, num_variants=2):
    """Create contrast examples with proper label updates (for comprehensive analysis)."""
    contrast_pairs = []
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    original_label = example["label"]
    
    # Strategy 1: Add negation
    if "not" not in hypothesis.lower() and "is" in hypothesis.lower():
        negated_hyp = hypothesis.replace(" is ", " is not ", 1)
        if negated_hyp != hypothesis:
            new_label = 2 if original_label == 0 else (0 if original_label == 2 else original_label)
            contrast_pairs.append({
                "premise": premise,
                "hypothesis": negated_hyp,
                "label": new_label,
                "contrast_type": "negation_added",
                "original_hypothesis": hypothesis,
                "original_label": original_label
            })
    
    # Strategy 2: Antonym replacement
    replacements = [
        ("happy", "sad", 2),
        ("big", "small", 2),
        ("hot", "cold", 2),
        ("good", "bad", 2),
        ("running", "walking", 2),
        ("preparing", "eating", 2),
    ]
    
    for old_word, new_word, likely_label in replacements:
        if old_word in hypothesis.lower() and len(contrast_pairs) < num_variants:
            modified_hyp = re.sub(rf"\b{old_word}\b", new_word, hypothesis, flags=re.IGNORECASE)
            if modified_hyp != hypothesis:
                new_label = likely_label if original_label == 0 else original_label
                contrast_pairs.append({
                    "premise": premise,
                    "hypothesis": modified_hyp,
                    "label": new_label,
                    "contrast_type": f"antonym_{old_word}_{new_word}",
                    "original_hypothesis": hypothesis,
                    "original_label": original_label
                })
                break
    
    return contrast_pairs


def evaluate_contrast_sets(model, tokenizer, original_dataset, device="cuda", num_examples=50):
    """Create and evaluate contrast sets."""
    print("\n[APPROACH 1] Creating and evaluating contrast sets...")
    
    sample_indices = random.sample(range(len(original_dataset)), min(num_examples, len(original_dataset)))
    sample_dataset = original_dataset.select(sample_indices)
    
    contrast_examples = []
    original_examples = []
    
    for ex in sample_dataset:
        original_examples.append(ex)
        contrasts = create_contrast_examples(ex, num_variants=2)
        contrast_examples.extend(contrasts)
    
    print(f"  Created {len(contrast_examples)} contrast examples from {len(original_examples)} originals")
    
    orig_acc = evaluate_slice_simple(model, tokenizer, original_examples, device)
    contrast_acc = evaluate_slice_simple(model, tokenizer, contrast_examples, device)
    
    return {
        "original_accuracy": orig_acc,
        "contrast_accuracy": contrast_acc,
        "performance_drop": orig_acc - contrast_acc,
        "num_original": len(original_examples),
        "num_contrast": len(contrast_examples)
    }


def evaluate_contrast_pairs_with_logits(model, tokenizer, dataset, device="cuda", num_pairs=10):
    """Evaluate contrast pairs and return logits/probabilities."""
    print("\n[ANALYSIS 3] Creating and evaluating contrast pairs with logits...")
    
    sample_indices = random.sample(range(len(dataset)), min(num_pairs * 2, len(dataset)))
    sample_dataset = dataset.select(sample_indices)
    
    contrast_pairs_data = []
    
    for ex in sample_dataset:
        pairs = create_contrast_pairs(ex, num_variants=1)
        if pairs and len(contrast_pairs_data) < num_pairs:
            pair = pairs[0]
            contrast_pairs_data.append({
                "original": ex,
                "contrast": pair
            })
    
    print(f"  Created {len(contrast_pairs_data)} contrast pairs")
    
    if len(contrast_pairs_data) == 0:
        return []
    
    original_list = [pair["original"] for pair in contrast_pairs_data]
    contrast_list = [pair["contrast"] for pair in contrast_pairs_data]
    
    orig_results = evaluate_with_logits(model, tokenizer, original_list, device=device)
    contrast_results = evaluate_with_logits(model, tokenizer, contrast_list, device=device)
    
    contrast_pairs_results = []
    for i, pair_data in enumerate(contrast_pairs_data):
        orig_ex = pair_data["original"]
        contrast_ex = pair_data["contrast"]
        
        orig_logits = orig_results["logits"][i]
        orig_probs = orig_results["probabilities"][i]
        orig_pred = orig_results["predictions"][i]
        orig_gold = orig_results["gold_labels"][i]
        
        contrast_logits = contrast_results["logits"][i]
        contrast_probs = contrast_results["probabilities"][i]
        contrast_pred = contrast_results["predictions"][i]
        contrast_gold = contrast_results["gold_labels"][i]
        
        label_names = ["entailment", "neutral", "contradiction"]
        
        contrast_pairs_results.append({
            "premise": orig_ex["premise"],
            "original_hypothesis": orig_ex["hypothesis"],
            "original_label": label_names[orig_gold],
            "original_prediction": label_names[orig_pred],
            "original_logits": orig_logits.tolist(),
            "original_probabilities": orig_probs.tolist(),
            "contrast_hypothesis": contrast_ex["hypothesis"],
            "contrast_label": label_names[contrast_gold],
            "contrast_prediction": label_names[contrast_pred],
            "contrast_logits": contrast_logits.tolist(),
            "contrast_probabilities": contrast_probs.tolist(),
            "contrast_type": contrast_ex.get("contrast_type", "unknown")
        })
    
    return contrast_pairs_results


# ============================================================================
# APPROACH 2: MODEL ABLATIONS - Hypothesis-Only Baseline
# ============================================================================

def evaluate_hypothesis_only(model, tokenizer, dataset, device="cuda", batch_size=32):
    """Evaluate model using only hypothesis (ignoring premise)."""
    print("\n[APPROACH 2] Evaluating hypothesis-only baseline...")
    
    model.eval()
    predictions = []
    gold_labels = []
    
    dataset_list = [dataset[i] for i in range(len(dataset))]
    
    for i in tqdm(range(0, len(dataset_list), batch_size), desc="Hypothesis-only"):
        batch = dataset_list[i:i+batch_size]
        premises = [""] * len(batch)
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
# APPROACH 3: STATISTICAL TESTS - N-gram Correlations
# ============================================================================

def find_ngram_correlations(dataset, predictions, gold_labels, n=2, top_k=10):
    """Find spurious n-gram correlations with labels."""
    print(f"\n[APPROACH 3] Finding {n}-gram correlations with labels...")
    
    label_ngrams = {0: Counter(), 1: Counter(), 2: Counter()}
    dataset_list = [dataset[i] for i in range(len(dataset))]
    
    for ex, pred, gold in zip(dataset_list, predictions, gold_labels):
        hypothesis = ex["hypothesis"].lower()
        words = hypothesis.split()
        
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            label_ngrams[gold][ngram] += 1
    
    correlations = []
    
    for label in [0, 1, 2]:
        label_name = ["entailment", "neutral", "contradiction"][label]
        all_ngrams = set()
        for l in [0, 1, 2]:
            all_ngrams.update(label_ngrams[l].keys())
        
        for ngram in all_ngrams:
            count_in_label = label_ngrams[label][ngram]
            total_count = sum(label_ngrams[l][ngram] for l in [0, 1, 2])
            
            if total_count >= 5:
                correlation = count_in_label / total_count if total_count > 0 else 0
                if correlation > 0.7:
                    correlations.append({
                        "ngram": ngram,
                        "label": label_name,
                        "correlation": correlation,
                        "count_in_label": count_in_label,
                        "total_count": total_count
                    })
    
    correlations.sort(key=lambda x: x["correlation"], reverse=True)
    return correlations[:top_k]


# ============================================================================
# Comprehensive Analysis Functions
# ============================================================================

def evaluate_by_hypothesis_length(model, tokenizer, dataset, device="cuda"):
    """Evaluate model on slices defined by hypothesis length."""
    print("\n[ANALYSIS 1] Evaluating by hypothesis length...")
    
    dataset_with_length = dataset.map(lambda x: {"hyp_length": get_hypothesis_length(x)})
    
    length_slices = {
        "hyp_1_4": dataset_with_length.filter(lambda x: 1 <= x["hyp_length"] <= 4),
        "hyp_5_9": dataset_with_length.filter(lambda x: 5 <= x["hyp_length"] <= 9),
        "hyp_10_plus": dataset_with_length.filter(lambda x: x["hyp_length"] >= 10)
    }
    
    results = {}
    for slice_name, slice_data in length_slices.items():
        if len(slice_data) > 0:
            print(f"  Evaluating {slice_name} ({len(slice_data)} examples)...")
            slice_results = evaluate_on_slice(model, tokenizer, slice_data, device=device)
            results[slice_name] = {
                "accuracy": slice_results["accuracy"],
                "num_examples": slice_results["num_examples"]
            }
            print(f"    Accuracy: {slice_results['accuracy']:.4f}")
    
    return results


def analyze_negation_misclassification(model, tokenizer, dataset, device="cuda"):
    """Analyze misclassification patterns for examples with negation in hypothesis."""
    print("\n[ANALYSIS 2] Analyzing negation-triggered misclassification...")
    
    neg_hyp_dataset = dataset.filter(has_negation_in_hypothesis)
    print(f"  Found {len(neg_hyp_dataset)} examples with negation in hypothesis")
    
    if len(neg_hyp_dataset) == 0:
        return {
            "num_negation_examples": 0,
            "contradiction_predictions_when_gold_not_contradiction": 0,
            "misclassification_rate": 0.0
        }
    
    results = evaluate_on_slice(model, tokenizer, neg_hyp_dataset, device=device)
    
    predictions = results["predictions"]
    gold_labels = results["gold_labels"]
    
    contradiction_label = 2
    count_contradiction_when_not = sum(
        1 for pred, gold in zip(predictions, gold_labels)
        if pred == contradiction_label and gold != contradiction_label
    )
    
    misclassification_rate = count_contradiction_when_not / len(predictions) if len(predictions) > 0 else 0.0
    
    print(f"  Contradiction predictions when gold ≠ contradiction: {count_contradiction_when_not}/{len(predictions)}")
    print(f"  Misclassification rate: {misclassification_rate:.4f}")
    
    return {
        "num_negation_examples": len(neg_hyp_dataset),
        "contradiction_predictions_when_gold_not_contradiction": count_contradiction_when_not,
        "total_predictions": len(predictions),
        "misclassification_rate": misclassification_rate
    }


# ============================================================================
# Model Detection
# ============================================================================

def find_model_paths():
    """Automatically detect available models (baseline and/or contrast-augmented)."""
    baseline_paths = [
        "./trained_snli_baseline",
        "./results/baseline",
        "../trained_snli_baseline",
    ]
    
    contrast_paths = [
        "./trained_snli_contrast",
        "./results/contrast",
        "../trained_snli_contrast",
    ]
    
    def check_path(path):
        """Check if path contains a valid model."""
        if not os.path.exists(path):
            return False
        if not os.path.exists(os.path.join(path, "config.json")):
            return False
        # Check for model weights
        has_weights = (
            os.path.exists(os.path.join(path, "model.safetensors")) or
            os.path.exists(os.path.join(path, "pytorch_model.bin")) or
            os.path.exists(os.path.join(path, "checkpoint-99500", "model.safetensors")) or
            os.path.exists(os.path.join(path, "checkpoint-50000", "model.safetensors"))
        )
        return has_weights
    
    baseline_path = None
    contrast_path = None
    
    for path in baseline_paths:
        if check_path(path):
            baseline_path = path
            break
    
    for path in contrast_paths:
        if check_path(path):
            contrast_path = path
            break
    
    return baseline_path, contrast_path


def load_model(model_path, device="cuda"):
    """Load model and tokenizer from path."""
    if not model_path or not os.path.exists(model_path):
        print(f"[WARNING] Model not found at {model_path}, using pre-trained ELECTRA-small")
        model_path = "google/electra-small-discriminator"
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer_path = model_path if os.path.exists(os.path.join(model_path, "tokenizer.json")) else model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer
    except Exception as e:
        print(f"[ERROR] Error loading model from {model_path}: {e}")
        print("   Trying pre-trained model as fallback...")
        model_path = "google/electra-small-discriminator"
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer


# ============================================================================
# Main Analysis Function
# ============================================================================

def run_full_analysis(model_path, model_name="baseline", device="cuda", include_attention=False):
    """Run complete analysis for a single model."""
    print("=" * 60)
    print(f"Complete Analysis: {model_name.upper()}")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading SNLI dataset...")
    dataset = load_dataset("snli")
    val_dataset = dataset["validation"].filter(lambda x: x["label"] != -1)
    print(f"[OK] Validation set: {len(val_dataset)} examples")
    
    # Load model
    print(f"\n2. Loading {model_name} model...")
    model, tokenizer = load_model(model_path, device=device)
    print(f"[OK] Model loaded on {device}")
    
    results = {}
    
    # ========================================================================
    # BASIC SLICE ANALYSIS
    # ========================================================================
    print("\n3. Creating dataset slices...")
    slices = create_slices(val_dataset)
    
    print("\n4. Evaluating model on slices...")
    print("  Evaluating on full validation set...")
    full_results = evaluate_on_slice(model, tokenizer, val_dataset, device=device)
    results["full_validation"] = full_results
    print(f"    Accuracy: {full_results['accuracy']:.4f}")
    
    results["slice_results"] = {}
    for slice_name, slice_data in slices.items():
        print(f"  Evaluating on {slice_name} slice ({len(slice_data)} examples)...")
        slice_results = evaluate_on_slice(model, tokenizer, slice_data, device=device)
        results["slice_results"][slice_name] = slice_results
        print(f"    Accuracy: {slice_results['accuracy']:.4f}")
    
    # Collect error examples
    print("\n5. Collecting error examples...")
    error_examples = {}
    for slice_name in ["negation", "long_premise", "high_overlap"]:
        if slice_name in slices and slice_name in results["slice_results"]:
            errors = collect_error_examples(
                slices[slice_name],
                results["slice_results"][slice_name]["predictions"],
                results["slice_results"][slice_name]["gold_labels"],
                num_examples=10
            )
            error_examples[slice_name] = errors
            print(f"  [OK] Collected {len(errors)} error examples from {slice_name} slice")
    
    results["error_examples"] = error_examples
    
    # ========================================================================
    # ENHANCED ANALYSIS (Contrast Sets, Hypothesis-Only, N-grams)
    # ========================================================================
    print("\n6. Enhanced analysis (contrast sets, hypothesis-only, n-grams)...")
    
    # Contrast Sets
    contrast_results = evaluate_contrast_sets(model, tokenizer, val_dataset, device=device, num_examples=30)
    results["contrast_sets"] = contrast_results
    print(f"\n  Contrast Sets Results:")
    print(f"    Original accuracy: {contrast_results['original_accuracy']:.4f}")
    print(f"    Contrast accuracy: {contrast_results['contrast_accuracy']:.4f}")
    print(f"    Performance drop: {contrast_results['performance_drop']:.4f}")
    
    # Hypothesis-Only Baseline
    hypo_results = evaluate_hypothesis_only(model, tokenizer, val_dataset, device=device)
    results["hypothesis_only"] = hypo_results
    print(f"\n  Hypothesis-Only Results:")
    print(f"    Accuracy: {hypo_results['hypothesis_only_accuracy']:.4f}")
    print(f"    (If high, model may rely on hypothesis-only heuristics)")
    
    # N-gram Correlations
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
    
    # ========================================================================
    # COMPREHENSIVE ANALYSIS (Hypothesis Length, Negation Misclassification, Logits)
    # ========================================================================
    print("\n7. Comprehensive analysis (hypothesis length, negation misclassification, logits)...")
    
    # Hypothesis length slices
    results["hypothesis_length"] = evaluate_by_hypothesis_length(model, tokenizer, val_dataset, device=device)
    
    # Negation misclassification
    results["negation_misclassification"] = analyze_negation_misclassification(model, tokenizer, val_dataset, device=device)
    
    # Contrast pairs with logits
    results["contrast_pairs"] = evaluate_contrast_pairs_with_logits(model, tokenizer, val_dataset, device=device, num_pairs=10)
    
    # Attention heatmaps (optional)
    if include_attention:
        print("\n[SKIP] Attention heatmaps (not implemented in unified version)")
        results["attention_heatmaps"] = []
    else:
        results["attention_heatmaps"] = []
        print("\n[SKIP] Attention heatmaps (set include_attention=True to enable)")
    
    return results


# ============================================================================
# Comparison Function
# ============================================================================

def compare_before_after(baseline_results, contrast_results):
    """Compare baseline vs contrast-augmented model."""
    print("\n" + "=" * 60)
    print("BEFORE/AFTER COMPARISON")
    print("=" * 60)
    
    comparison = {}
    
    # Overall accuracy
    if "full_validation" in baseline_results and "full_validation" in contrast_results:
        baseline_acc = baseline_results["full_validation"]["accuracy"]
        contrast_acc = contrast_results["full_validation"]["accuracy"]
        comparison["overall"] = {
            "baseline": baseline_acc,
            "contrast_augmented": contrast_acc,
            "change": contrast_acc - baseline_acc
        }
        print(f"\nOverall Accuracy: {baseline_acc:.4f} → {contrast_acc:.4f} (Δ={contrast_acc-baseline_acc:+.4f})")
    
    # Negation slice
    if "slice_results" in baseline_results and "negation" in baseline_results["slice_results"]:
        if "slice_results" in contrast_results and "negation" in contrast_results["slice_results"]:
            baseline_acc = baseline_results["slice_results"]["negation"]["accuracy"]
            contrast_acc = contrast_results["slice_results"]["negation"]["accuracy"]
            comparison["negation"] = {
                "baseline": baseline_acc,
                "contrast_augmented": contrast_acc,
                "change": contrast_acc - baseline_acc
            }
            print(f"Negation slice: {baseline_acc:.4f} → {contrast_acc:.4f} (Δ={contrast_acc-baseline_acc:+.4f})")
    
    # Hypothesis-only
    if "hypothesis_only" in baseline_results and "hypothesis_only" in contrast_results:
        baseline_acc = baseline_results["hypothesis_only"]["hypothesis_only_accuracy"]
        contrast_acc = contrast_results["hypothesis_only"]["hypothesis_only_accuracy"]
        comparison["hypothesis_only"] = {
            "baseline": baseline_acc,
            "contrast_augmented": contrast_acc,
            "change": contrast_acc - baseline_acc
        }
        print(f"Hypothesis-only: {baseline_acc:.4f} → {contrast_acc:.4f} (Δ={contrast_acc-baseline_acc:+.4f})")
    
    # Contrast sets
    if "contrast_sets" in baseline_results and "contrast_sets" in contrast_results:
        baseline_drop = baseline_results["contrast_sets"]["performance_drop"]
        contrast_drop = contrast_results["contrast_sets"]["performance_drop"]
        comparison["contrast_sets"] = {
            "baseline_drop": baseline_drop,
            "contrast_augmented_drop": contrast_drop,
            "improvement": baseline_drop - contrast_drop
        }
        print(f"Contrast sets drop: {baseline_drop:.4f} → {contrast_drop:.4f} (improvement: {baseline_drop-contrast_drop:.4f})")
    
    # Hypothesis length
    if "hypothesis_length" in baseline_results and "hypothesis_length" in contrast_results:
        comparison["hypothesis_length"] = {}
        for length_bucket in ["hyp_1_4", "hyp_5_9", "hyp_10_plus"]:
            if length_bucket in baseline_results["hypothesis_length"] and length_bucket in contrast_results["hypothesis_length"]:
                baseline_acc = baseline_results["hypothesis_length"][length_bucket]["accuracy"]
                contrast_acc = contrast_results["hypothesis_length"][length_bucket]["accuracy"]
                comparison["hypothesis_length"][length_bucket] = {
                    "baseline": baseline_acc,
                    "contrast_augmented": contrast_acc,
                    "change": contrast_acc - baseline_acc
                }
                print(f"{length_bucket}: {baseline_acc:.4f} → {contrast_acc:.4f} (Δ={contrast_acc-baseline_acc:+.4f})")
    
    return comparison


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point - automatically detects and analyzes available models."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified artifact analysis for Part 1 and Part 2")
    parser.add_argument("--baseline_model", type=str, default=None,
                       help="Path to baseline model (auto-detected if not specified)")
    parser.add_argument("--contrast_model", type=str, default=None,
                       help="Path to contrast-augmented model (auto-detected if not specified)")
    parser.add_argument("--include_attention", action="store_true",
                       help="Include attention heatmaps (may be slow)")
    parser.add_argument("--model_only", type=str, choices=["baseline", "contrast"],
                       help="Run analysis for only one model")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Auto-detect models if not specified
    if args.baseline_model is None or args.contrast_model is None:
        detected_baseline, detected_contrast = find_model_paths()
        if args.baseline_model is None:
            args.baseline_model = detected_baseline
        if args.contrast_model is None:
            args.contrast_model = detected_contrast
    
    baseline_results = None
    contrast_results = None
    
    # Run baseline analysis
    if args.model_only != "contrast" and args.baseline_model:
        baseline_results = run_full_analysis(
            args.baseline_model,
            model_name="baseline",
            device=device,
            include_attention=args.include_attention
        )
        
        # Save baseline results
        output_file = "analysis_results_baseline.json"
        with open(output_file, "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"\n[OK] Saved baseline results to {output_file}")
        
        # Save basic results (for backward compatibility)
        basic_output = {
            "full_validation_accuracy": baseline_results["full_validation"]["accuracy"],
            "slice_results": {
                name: {
                    "accuracy": res["accuracy"],
                    "num_examples": res["num_examples"]
                }
                for name, res in baseline_results["slice_results"].items()
            },
            "error_examples": baseline_results["error_examples"]
        }
        with open("analysis_results.json", "w") as f:
            json.dump(basic_output, f, indent=2)
        
        # Save enhanced results (for backward compatibility)
        enhanced_output = {
            "full_validation_accuracy": baseline_results["full_validation"]["accuracy"],
            "slice_results": {
                name: {"accuracy": res["accuracy"], "num_examples": res["num_examples"]}
                for name, res in baseline_results["slice_results"].items()
            },
            "contrast_sets": baseline_results["contrast_sets"],
            "hypothesis_only": baseline_results["hypothesis_only"],
            "ngram_correlations": baseline_results["ngram_correlations"]
        }
        with open("analysis_results_enhanced.json", "w") as f:
            json.dump(enhanced_output, f, indent=2)
        
        # Save comprehensive results
        comprehensive_output = {
            "hypothesis_length": baseline_results["hypothesis_length"],
            "negation_misclassification": baseline_results["negation_misclassification"],
            "contrast_pairs": baseline_results["contrast_pairs"],
            "attention_heatmaps": baseline_results["attention_heatmaps"],
            "slice_results": {
                "negation": baseline_results["slice_results"]["negation"]
            },
            "hypothesis_only": baseline_results["hypothesis_only"]
        }
        with open("comprehensive_analysis_baseline.json", "w") as f:
            json.dump(comprehensive_output, f, indent=2)
        
        # Create visualization
        visualize_results(baseline_results["slice_results"], output_path="analysis_results.png")
    
    # Run contrast-augmented analysis
    if args.model_only != "baseline" and args.contrast_model:
        contrast_results = run_full_analysis(
            args.contrast_model,
            model_name="contrast-augmented",
            device=device,
            include_attention=args.include_attention
        )
        
        # Save contrast results
        output_file = "analysis_results_contrast.json"
        with open(output_file, "w") as f:
            json.dump(contrast_results, f, indent=2)
        print(f"\n[OK] Saved contrast-augmented results to {output_file}")
        
        # Save comprehensive results
        comprehensive_output = {
            "hypothesis_length": contrast_results["hypothesis_length"],
            "negation_misclassification": contrast_results["negation_misclassification"],
            "contrast_pairs": contrast_results["contrast_pairs"],
            "attention_heatmaps": contrast_results["attention_heatmaps"],
            "slice_results": {
                "negation": contrast_results["slice_results"]["negation"]
            },
            "hypothesis_only": contrast_results["hypothesis_only"]
        }
        with open("comprehensive_analysis_contrast.json", "w") as f:
            json.dump(comprehensive_output, f, indent=2)
    
    # Compare before/after if both models analyzed
    if baseline_results and contrast_results:
        comparison = compare_before_after(baseline_results, contrast_results)
        
        comparison_file = "comprehensive_analysis_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n[OK] Saved comparison to {comparison_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)
    
    if baseline_results:
        print(f"\nBaseline Results:")
        print(f"  Overall accuracy: {baseline_results['full_validation']['accuracy']:.4f}")
        print(f"  Hypothesis-only: {baseline_results['hypothesis_only']['hypothesis_only_accuracy']:.4f}")
        print(f"  Contrast sets drop: {baseline_results['contrast_sets']['performance_drop']:.4f}")
    
    if contrast_results:
        print(f"\nContrast-Augmented Results:")
        print(f"  Overall accuracy: {contrast_results['full_validation']['accuracy']:.4f}")
        print(f"  Hypothesis-only: {contrast_results['hypothesis_only']['hypothesis_only_accuracy']:.4f}")
        print(f"  Contrast sets drop: {contrast_results['contrast_sets']['performance_drop']:.4f}")
    
    if baseline_results and contrast_results:
        print(f"\nImprovement:")
        improvement = baseline_results['contrast_sets']['performance_drop'] - contrast_results['contrast_sets']['performance_drop']
        print(f"  Contrast sets drop reduced by: {improvement:.4f}")


if __name__ == "__main__":
    main()


"""
Comprehensive Analysis Script for Part 1 and Part 2
Collects all metrics needed for enhanced writeup:
1. Slice accuracy by hypothesis length
2. Negation-triggered misclassification rate
3. Logits/softmax probabilities for contrast pairs
4. Attention heatmaps (optional)
5. Before/after comparison on key slices
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

def has_negation_in_hypothesis(example):
    """Check if hypothesis contains negation words."""
    hypothesis = example.get("hypothesis", "").lower()
    negation_words = r"\b(not|no|never|nobody|nothing|none|neither|nowhere|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b"
    return bool(re.search(negation_words, hypothesis))


def has_negation(example):
    """Check if example contains negation words (premise or hypothesis)."""
    text = (example.get("premise", "") + " " + example.get("hypothesis", "")).lower()
    negation_words = r"\b(not|no|never|nobody|nothing|none|neither|nowhere|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b"
    return bool(re.search(negation_words, text))


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

def evaluate_with_logits(model, tokenizer, examples_list, device="cuda", batch_size=32):
    """
    Evaluate model and return predictions, gold labels, and logits/softmax probabilities.
    """
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


# ============================================================================
# 1. Hypothesis Length Slices
# ============================================================================

def evaluate_by_hypothesis_length(model, tokenizer, dataset, device="cuda"):
    """Evaluate model on slices defined by hypothesis length."""
    print("\n[ANALYSIS 1] Evaluating by hypothesis length...")
    
    # Add length information to dataset
    dataset_with_length = dataset.map(lambda x: {"hyp_length": get_hypothesis_length(x)})
    
    # Define length buckets
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


# ============================================================================
# 2. Negation-Triggered Misclassification
# ============================================================================

def analyze_negation_misclassification(model, tokenizer, dataset, device="cuda"):
    """Analyze misclassification patterns for examples with negation in hypothesis."""
    print("\n[ANALYSIS 2] Analyzing negation-triggered misclassification...")
    
    # Filter examples with negation in hypothesis
    neg_hyp_dataset = dataset.filter(has_negation_in_hypothesis)
    print(f"  Found {len(neg_hyp_dataset)} examples with negation in hypothesis")
    
    if len(neg_hyp_dataset) == 0:
        return {
            "num_negation_examples": 0,
            "contradiction_predictions_when_gold_not_contradiction": 0,
            "misclassification_rate": 0.0
        }
    
    # Evaluate on this slice
    results = evaluate_on_slice(model, tokenizer, neg_hyp_dataset, device=device)
    
    predictions = results["predictions"]
    gold_labels = results["gold_labels"]
    
    # Count: contradiction predictions when gold ≠ contradiction
    contradiction_label = 2  # Assuming 0=entailment, 1=neutral, 2=contradiction
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
# 3. Contrast Pairs with Logits/Probabilities
# ============================================================================

def create_contrast_pairs(example, num_variants=2):
    """Create contrast examples with proper label updates."""
    contrast_pairs = []
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    original_label = example["label"]
    
    # Strategy 1: Add negation
    if "not" not in hypothesis.lower() and "is" in hypothesis.lower():
        negated_hyp = hypothesis.replace(" is ", " is not ", 1)
        if negated_hyp != hypothesis:
            # Adding "not" typically flips entailment to contradiction
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
        ("happy", "sad", 2),  # (old, new, likely new label if was entailment)
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
                # If original was entailment, antonym likely makes it contradiction
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


def evaluate_contrast_pairs_with_logits(model, tokenizer, dataset, device="cuda", num_pairs=10):
    """Evaluate contrast pairs and return logits/probabilities."""
    print("\n[ANALYSIS 3] Creating and evaluating contrast pairs with logits...")
    
    # Sample examples to create contrasts
    sample_indices = random.sample(range(len(dataset)), min(num_pairs * 2, len(dataset)))
    sample_dataset = dataset.select(sample_indices)
    
    contrast_pairs_data = []
    original_examples = []
    
    for ex in sample_dataset:
        pairs = create_contrast_pairs(ex, num_variants=1)
        if pairs and len(contrast_pairs_data) < num_pairs:
            pair = pairs[0]
            contrast_pairs_data.append({
                "original": ex,
                "contrast": pair
            })
            original_examples.append(ex)
    
    print(f"  Created {len(contrast_pairs_data)} contrast pairs")
    
    # Evaluate originals and contrasts with logits
    original_list = [pair["original"] for pair in contrast_pairs_data]
    contrast_list = [pair["contrast"] for pair in contrast_pairs_data]
    
    orig_results = evaluate_with_logits(model, tokenizer, original_list, device=device)
    contrast_results = evaluate_with_logits(model, tokenizer, contrast_list, device=device)
    
    # Package results
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
# 4. Attention Heatmaps (Optional)
# ============================================================================

def get_attention_weights(model, tokenizer, premise, hypothesis, device="cuda", layer_idx=-1):
    """
    Extract attention weights from model.
    Note: This requires model to output attentions. May not work with all models.
    """
    try:
        inputs = tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # Tuple of attention tensors
        
        if attentions and len(attentions) > 0:
            # Get attention from specified layer (default: last layer)
            attention = attentions[layer_idx]  # Shape: (batch, heads, seq_len, seq_len)
            # Average over heads
            attention = attention.mean(dim=1).squeeze(0).cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            return attention, tokens
        else:
            return None, None
    except Exception as e:
        print(f"  [WARNING] Could not extract attention: {e}")
        return None, None


def create_attention_heatmaps(model, tokenizer, examples, device="cuda", num_examples=5):
    """Create attention heatmaps for selected examples."""
    print("\n[ANALYSIS 4] Creating attention heatmaps (optional)...")
    
    attention_results = []
    sample_examples = random.sample(examples, min(num_examples, len(examples)))
    
    for i, ex in enumerate(sample_examples):
        premise = ex["premise"]
        hypothesis = ex["hypothesis"]
        
        attention, tokens = get_attention_weights(model, tokenizer, premise, hypothesis, device=device)
        
        if attention is not None:
            attention_results.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "tokens": tokens,
                "attention_matrix": attention.tolist()
            })
            print(f"  [OK] Extracted attention for example {i+1}")
        else:
            print(f"  [SKIP] Could not extract attention for example {i+1}")
    
    return attention_results


# ============================================================================
# 5. Before/After Comparison
# ============================================================================

def compare_before_after(baseline_results, contrast_results, key_slices=["negation", "hypothesis_only", "length"]):
    """Compare baseline vs contrast-augmented model on key slices."""
    print("\n[ANALYSIS 5] Comparing before/after mitigation...")
    
    comparison = {}
    
    # Negation slice
    if "negation" in baseline_results.get("slice_results", {}) and "negation" in contrast_results.get("slice_results", {}):
        baseline_acc = baseline_results["slice_results"]["negation"]["accuracy"]
        contrast_acc = contrast_results["slice_results"]["negation"]["accuracy"]
        comparison["negation"] = {
            "baseline": baseline_acc,
            "contrast_augmented": contrast_acc,
            "change": contrast_acc - baseline_acc
        }
        print(f"  Negation slice: {baseline_acc:.4f} → {contrast_acc:.4f} (Δ={contrast_acc-baseline_acc:+.4f})")
    
    # Hypothesis-only
    if "hypothesis_only" in baseline_results and "hypothesis_only" in contrast_results:
        baseline_acc = baseline_results["hypothesis_only"]["hypothesis_only_accuracy"]
        contrast_acc = contrast_results["hypothesis_only"]["hypothesis_only_accuracy"]
        comparison["hypothesis_only"] = {
            "baseline": baseline_acc,
            "contrast_augmented": contrast_acc,
            "change": contrast_acc - baseline_acc
        }
        print(f"  Hypothesis-only: {baseline_acc:.4f} → {contrast_acc:.4f} (Δ={contrast_acc-baseline_acc:+.4f})")
    
    # Length slices (hypothesis length)
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
                print(f"  {length_bucket}: {baseline_acc:.4f} → {contrast_acc:.4f} (Δ={contrast_acc-baseline_acc:+.4f})")
    
    return comparison


# ============================================================================
# Main Analysis Function
# ============================================================================

def run_comprehensive_analysis(model_path, model_name="baseline", device="cuda", include_attention=False):
    """Run comprehensive analysis for a single model."""
    print("=" * 60)
    print(f"Comprehensive Analysis: {model_name.upper()}")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading SNLI dataset...")
    dataset = load_dataset("snli")
    val_dataset = dataset["validation"].filter(lambda x: x["label"] != -1)
    print(f"[OK] Validation set: {len(val_dataset)} examples")
    
    # Load model
    print("\n2. Loading model...")
    if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"[WARNING] Model not found at {model_path}, using pre-trained ELECTRA-small")
        model_path = "google/electra-small-discriminator"
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer_path = model_path if os.path.exists(os.path.join(model_path, "tokenizer.json")) else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"[OK] Model loaded on {device}")
    
    results = {}
    
    # 1. Hypothesis length slices
    results["hypothesis_length"] = evaluate_by_hypothesis_length(model, tokenizer, val_dataset, device=device)
    
    # 2. Negation misclassification
    results["negation_misclassification"] = analyze_negation_misclassification(model, tokenizer, val_dataset, device=device)
    
    # 3. Contrast pairs with logits
    results["contrast_pairs"] = evaluate_contrast_pairs_with_logits(model, tokenizer, val_dataset, device=device, num_pairs=10)
    
    # 4. Attention heatmaps (optional)
    if include_attention:
        dataset_list = [val_dataset[i] for i in range(min(100, len(val_dataset)))]
        results["attention_heatmaps"] = create_attention_heatmaps(model, tokenizer, dataset_list, device=device, num_examples=5)
    else:
        results["attention_heatmaps"] = []
        print("\n[SKIP] Attention heatmaps (set include_attention=True to enable)")
    
    # 5. Standard slices (negation, hypothesis-only, etc.)
    print("\n5. Evaluating standard slices...")
    
    # Negation slice
    negation_slice = val_dataset.filter(has_negation)
    if len(negation_slice) > 0:
        results["slice_results"] = {}
        results["slice_results"]["negation"] = evaluate_on_slice(model, tokenizer, negation_slice, device=device)
        print(f"  Negation slice accuracy: {results['slice_results']['negation']['accuracy']:.4f}")
    
    # Hypothesis-only baseline
    print("\n6. Evaluating hypothesis-only baseline...")
    hypothesis_only_results = evaluate_hypothesis_only(model, tokenizer, val_dataset, device=device)
    results["hypothesis_only"] = hypothesis_only_results
    print(f"  Hypothesis-only accuracy: {hypothesis_only_results['hypothesis_only_accuracy']:.4f}")
    
    return results


def evaluate_hypothesis_only(model, tokenizer, dataset, device="cuda", batch_size=32):
    """Evaluate hypothesis-only baseline."""
    model.eval()
    predictions = []
    gold_labels = []
    
    dataset_list = [dataset[i] for i in range(len(dataset))]
    
    for i in tqdm(range(0, len(dataset_list), batch_size), desc="Hypothesis-only"):
        batch = dataset_list[i:i+batch_size]
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
# Main Entry Point
# ============================================================================

def main():
    """Run comprehensive analysis for both baseline and contrast-augmented models."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive artifact analysis")
    parser.add_argument("--baseline_model", type=str, default="./trained_snli_baseline",
                       help="Path to baseline model")
    parser.add_argument("--contrast_model", type=str, default="./trained_snli_contrast",
                       help="Path to contrast-augmented model")
    parser.add_argument("--include_attention", action="store_true",
                       help="Include attention heatmaps (may be slow)")
    parser.add_argument("--model_only", type=str, choices=["baseline", "contrast"],
                       help="Run analysis for only one model")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    baseline_results = None
    contrast_results = None
    
    # Run baseline analysis
    if args.model_only != "contrast":
        baseline_results = run_comprehensive_analysis(
            args.baseline_model,
            model_name="baseline",
            device=device,
            include_attention=args.include_attention
        )
        
        # Save baseline results
        output_file = "comprehensive_analysis_baseline.json"
        with open(output_file, "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"\n[OK] Saved baseline results to {output_file}")
    
    # Run contrast-augmented analysis
    if args.model_only != "baseline":
        contrast_results = run_comprehensive_analysis(
            args.contrast_model,
            model_name="contrast-augmented",
            device=device,
            include_attention=args.include_attention
        )
        
        # Save contrast results
        output_file = "comprehensive_analysis_contrast.json"
        with open(output_file, "w") as f:
            json.dump(contrast_results, f, indent=2)
        print(f"\n[OK] Saved contrast-augmented results to {output_file}")
    
    # Compare before/after if both models analyzed
    if baseline_results and contrast_results:
        comparison = compare_before_after(baseline_results, contrast_results)
        
        # Save comparison
        comparison_file = "comprehensive_analysis_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n[OK] Saved comparison to {comparison_file}")
    
    print("\n" + "=" * 60)
    print("✅ Comprehensive analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


"""
Script to extract and organize training data for Part 1 analysis.
Run this in Colab after training to extract useful data for analysis.
"""

import json
import os
from pathlib import Path


def extract_training_dynamics(trainer_state_path):
    """
    Extract training dynamics from trainer_state.json.
    Useful for dataset cartography (identifying hard/easy examples).
    """
    if not os.path.exists(trainer_state_path):
        print(f"[WARNING] {trainer_state_path} not found")
        return None
    
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    # Extract useful metrics
    training_dynamics = {
        "total_steps": trainer_state.get("max_steps", 0),
        "epochs": trainer_state.get("epoch", 0),
        "training_loss_history": trainer_state.get("log_history", []),
        "best_metric": trainer_state.get("best_metric", None),
        "best_model_checkpoint": trainer_state.get("best_model_checkpoint", None)
    }
    
    return training_dynamics


def extract_evaluation_predictions(eval_predictions_path):
    """
    Extract predictions from eval_predictions.jsonl.
    Returns list of examples with predictions for error analysis.
    """
    if not os.path.exists(eval_predictions_path):
        print(f"[WARNING] {eval_predictions_path} not found")
        return []
    
    predictions = []
    with open(eval_predictions_path, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    return predictions


def identify_hard_examples_from_predictions(predictions, top_k=100):
    """
    Identify hard examples based on prediction confidence.
    Low confidence = potentially hard examples.
    """
    hard_examples = []
    
    for ex in predictions:
        if "predicted_scores" in ex:
            scores = ex["predicted_scores"]
            predicted_label = ex["predicted_label"]
            gold_label = ex["label"]
            
            # Calculate confidence (max score - second max score)
            sorted_scores = sorted(scores, reverse=True)
            confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
            
            # Low confidence or wrong prediction = hard example
            is_wrong = predicted_label != gold_label
            is_low_confidence = confidence < 2.0  # Threshold
            
            if is_wrong or is_low_confidence:
                hard_examples.append({
                    **ex,
                    "confidence": confidence,
                    "is_error": is_wrong
                })
    
    # Sort by confidence (lowest first = hardest)
    hard_examples.sort(key=lambda x: x["confidence"])
    
    return hard_examples[:top_k]


def extract_slice_performance_from_predictions(predictions, slice_filter_func, slice_name):
    """
    Calculate performance on a specific slice using existing predictions.
    Useful when you don't have model weights but have predictions.
    """
    slice_examples = [ex for ex in predictions if slice_filter_func(ex)]
    
    if len(slice_examples) == 0:
        return None
    
    correct = sum(1 for ex in slice_examples if ex["predicted_label"] == ex["label"])
    accuracy = correct / len(slice_examples)
    
    return {
        "slice_name": slice_name,
        "accuracy": accuracy,
        "num_examples": len(slice_examples),
        "num_correct": correct,
        "num_errors": len(slice_examples) - correct
    }


def organize_data_for_analysis(model_dir="trained_snli_baseline", output_dir="analysis_data"):
    """
    Extract and organize all training data useful for analysis.
    """
    print("=" * 60)
    print("Extracting Training Data for Analysis")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extract training dynamics
    print("\n1. Extracting training dynamics...")
    trainer_state_path = os.path.join(model_dir, "trainer_state.json")
    training_dynamics = extract_training_dynamics(trainer_state_path)
    
    if training_dynamics:
        with open(os.path.join(output_dir, "training_dynamics.json"), 'w') as f:
            json.dump(training_dynamics, f, indent=2)
        print(f"   [OK] Saved training dynamics")
    else:
        print(f"   [WARNING] No training dynamics found")
    
    # 2. Extract evaluation predictions
    print("\n2. Extracting evaluation predictions...")
    eval_predictions_path = os.path.join(model_dir, "eval_predictions.jsonl")
    if not os.path.exists(eval_predictions_path):
        # Try alternative location
        eval_predictions_path = "eval_snli_baseline/eval_predictions.jsonl"
    
    predictions = extract_evaluation_predictions(eval_predictions_path)
    
    if predictions:
        with open(os.path.join(output_dir, "all_predictions.json"), 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"   [OK] Saved {len(predictions)} predictions")
    else:
        print(f"   [WARNING] No predictions found")
    
    # 3. Identify hard examples
    print("\n3. Identifying hard examples...")
    if predictions:
        hard_examples = identify_hard_examples_from_predictions(predictions, top_k=200)
        with open(os.path.join(output_dir, "hard_examples.json"), 'w') as f:
            json.dump(hard_examples, f, indent=2)
        print(f"   [OK] Identified {len(hard_examples)} hard examples")
        
        # Separate errors
        error_examples = [ex for ex in hard_examples if ex["is_error"]]
        with open(os.path.join(output_dir, "error_examples.json"), 'w') as f:
            json.dump(error_examples, f, indent=2)
        print(f"   [OK] Found {len(error_examples)} error examples")
    
    # 4. Calculate slice performance from predictions
    print("\n4. Calculating slice performance from predictions...")
    if predictions:
        slice_results = []
        
        # Negation slice
        def has_negation(ex):
            text = (ex.get("premise", "") + " " + ex.get("hypothesis", "")).lower()
            return bool(re.search(r"\b(not|no|never|nobody|nothing|none)\b", text))
        
        neg_result = extract_slice_performance_from_predictions(predictions, has_negation, "negation")
        if neg_result:
            slice_results.append(neg_result)
        
        # Long premise slice
        def has_long_premise(ex):
            premise = ex.get("premise", "")
            return len(premise.split()) >= 20
        
        long_result = extract_slice_performance_from_predictions(predictions, has_long_premise, "long_premise")
        if long_result:
            slice_results.append(long_result)
        
        if slice_results:
            with open(os.path.join(output_dir, "slice_performance.json"), 'w') as f:
                json.dump(slice_results, f, indent=2)
            print(f"   [OK] Calculated performance on {len(slice_results)} slices")
    
    # 5. Summary statistics
    print("\n5. Generating summary statistics...")
    if predictions:
        total = len(predictions)
        correct = sum(1 for ex in predictions if ex["predicted_label"] == ex["label"])
        accuracy = correct / total if total > 0 else 0
        
        # Error distribution by label
        error_by_label = {0: 0, 1: 0, 2: 0}
        for ex in predictions:
            if ex["predicted_label"] != ex["label"]:
                error_by_label[ex["label"]] = error_by_label.get(ex["label"], 0) + 1
        
        summary = {
            "total_examples": total,
            "overall_accuracy": accuracy,
            "num_correct": correct,
            "num_errors": total - correct,
            "error_distribution": {
                "entailment": error_by_label[0],
                "neutral": error_by_label[1],
                "contradiction": error_by_label[2]
            }
        }
        
        with open(os.path.join(output_dir, "summary_stats.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   [OK] Overall accuracy: {accuracy:.4f}")
    
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print(f"All data saved to: {output_dir}/")
    print("\nFiles created:")
    print("  - training_dynamics.json (training metrics)")
    print("  - all_predictions.json (all predictions)")
    print("  - hard_examples.json (low confidence/wrong predictions)")
    print("  - error_examples.json (wrong predictions only)")
    print("  - slice_performance.json (performance by slice)")
    print("  - summary_stats.json (overall statistics)")


if __name__ == "__main__":
    import re
    organize_data_for_analysis()


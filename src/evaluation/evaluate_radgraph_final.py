"""
RadGraph F1 Score Evaluation - CORRECT API
Based on discovery: forward(hyps) - single input, extracts entities
Then manually compares entities between predictions and references
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm


def calculate_radgraph_f1_correct(predictions, references, output_dir):
    """
    Calculate RadGraph F1 using correct API
    
    RadGraph extracts entities from text, then we compare them
    """
    from radgraph import RadGraph
    import torch
    
    print("\n" + "="*80)
    print("CALCULATING RADGRAPH F1 SCORES")
    print("="*80)
    
    # Initialize RadGraph
    print("\nInitializing RadGraph model...")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 4 else 'CPU'}")
    
    rg = RadGraph(reward_level="full", batch_size=8, cuda=device)
    print("✓ RadGraph model loaded")
    
    # Filter valid pairs
    valid_pairs = [
        (pred, ref) 
        for pred, ref in zip(predictions, references) 
        if pred and ref and len(pred.strip()) > 0 and len(ref.strip()) > 0
    ]
    
    print(f"\nValid report pairs: {len(valid_pairs)}")
    
    if len(valid_pairs) == 0:
        print("✗ No valid report pairs!")
        return None
    
    valid_preds = [p for p, _ in valid_pairs]
    valid_refs = [r for _, r in valid_pairs]
    
    print("\nExtracting clinical entities from predictions...")
    
    try:
        # Extract entities from predictions
        print("Processing predictions...")
        pred_graphs = rg(valid_preds)
        
        print("Processing references...")
        ref_graphs = rg(valid_refs)
        
        print(f"\n✓ Entity extraction complete!")
        print(f"Type of result: {type(pred_graphs)}")
        
        # Now we need to compare the graphs
        # RadGraph typically returns dictionaries with entity information
        
        if isinstance(pred_graphs, dict) and isinstance(ref_graphs, dict):
            # Both are dicts - likely indexed by sample
            print(f"Pred keys: {list(pred_graphs.keys())[:5]}...")
            print(f"Ref keys: {list(ref_graphs.keys())[:5]}...")
            
            # Calculate F1 for each sample
            f1_scores = []
            precision_scores = []
            recall_scores = []
            
            for i in range(len(valid_preds)):
                key = str(i) if str(i) in pred_graphs else i
                
                if key in pred_graphs and key in ref_graphs:
                    pred_graph = pred_graphs[key]
                    ref_graph = ref_graphs[key]
                    
                    # Extract entities
                    pred_entities = set()
                    ref_entities = set()
                    
                    # Parse graph structure
                    if isinstance(pred_graph, dict):
                        if 'entities' in pred_graph:
                            pred_entities = set(pred_graph['entities'].keys())
                        elif 'nodes' in pred_graph:
                            pred_entities = set(pred_graph['nodes'].keys())
                    
                    if isinstance(ref_graph, dict):
                        if 'entities' in ref_graph:
                            ref_entities = set(ref_graph['entities'].keys())
                        elif 'nodes' in ref_graph:
                            ref_entities = set(ref_graph['nodes'].keys())
                    
                    # Calculate F1
                    if len(ref_entities) == 0:
                        f1 = 1.0 if len(pred_entities) == 0 else 0.0
                        precision = 1.0 if len(pred_entities) == 0 else 0.0
                        recall = 1.0
                    else:
                        tp = len(pred_entities & ref_entities)
                        fp = len(pred_entities - ref_entities)
                        fn = len(ref_entities - pred_entities)
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    f1_scores.append(f1)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
            
            # Calculate averages
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
            avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
            
            print(f"\nResults:")
            print(f"  RadGraph F1:         {avg_f1:.4f}")
            print(f"  RadGraph Precision:  {avg_precision:.4f}")
            print(f"  RadGraph Recall:     {avg_recall:.4f}")
            
            # Save results
            results = {
                'radgraph_f1': avg_f1,
                'radgraph_precision': avg_precision,
                'radgraph_recall': avg_recall,
                'num_samples': len(valid_pairs),
                'per_sample_f1': f1_scores,
                'per_sample_precision': precision_scores,
                'per_sample_recall': recall_scores
            }
            
            output_path = Path(output_dir) / 'radgraph_scores.json'
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓ Results saved to: {output_path}")
            
            # Save report
            report_path = Path(output_dir) / 'radgraph_report.txt'
            with open(report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("RADGRAPH F1 EVALUATION REPORT\n")
                f.write("="*80 + "\n")
                f.write(f"Number of samples:     {len(valid_pairs)}\n")
                f.write(f"RadGraph F1 Score:     {avg_f1:.4f}\n")
                f.write(f"RadGraph Precision:    {avg_precision:.4f}\n")
                f.write(f"RadGraph Recall:       {avg_recall:.4f}\n")
                f.write("="*80 + "\n")
                if f1_scores:
                    import statistics
                    f.write(f"\nF1 Score Distribution:\n")
                    f.write(f"  Mean:   {avg_f1:.4f}\n")
                    f.write(f"  Min:    {min(f1_scores):.4f}\n")
                    f.write(f"  Max:    {max(f1_scores):.4f}\n")
                    if len(f1_scores) > 1:
                        f.write(f"  Median: {statistics.median(f1_scores):.4f}\n")
                        f.write(f"  StdDev: {statistics.stdev(f1_scores):.4f}\n")
                f.write("="*80 + "\n")
            
            print(f"✓ Report saved to: {report_path}")
            
            return results
        
        else:
            # Debug: print structure
            print(f"\n⚠️  Unexpected structure. Saving raw output for inspection...")
            
            debug_output = {
                'pred_graphs_type': str(type(pred_graphs)),
                'ref_graphs_type': str(type(ref_graphs)),
                'pred_graphs_sample': str(pred_graphs)[:500] if pred_graphs else None,
                'ref_graphs_sample': str(ref_graphs)[:500] if ref_graphs else None
            }
            
            debug_path = Path(output_dir) / 'radgraph_debug.json'
            with open(debug_path, 'w') as f:
                json.dump(debug_output, f, indent=2)
            
            print(f"✓ Debug info saved to: {debug_path}")
            print("Please share this file so we can fix the parsing!")
            
            return None
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Calculate RadGraph F1 Scores')
    parser.add_argument('--predictions_file', type=str, required=True,
                       help='Path to predictions.json file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit number of samples for testing')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"\nLoading predictions from: {args.predictions_file}")
    with open(args.predictions_file, 'r') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    references = data['references']
    
    if args.max_samples and args.max_samples < len(predictions):
        predictions = predictions[:args.max_samples]
        references = references[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")
    
    print(f"Total samples: {len(predictions)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate RadGraph F1
    results = calculate_radgraph_f1_correct(predictions, references, output_dir)
    
    if results:
        print("\n" + "="*80)
        print("✓ RADGRAPH EVALUATION COMPLETE!")
        print("="*80)
        print(f"RadGraph F1: {results['radgraph_f1']:.4f}")
        print(f"Samples:     {results['num_samples']}")
        print("="*80)
    else:
        print("\n⚠️  Evaluation completed but needs debugging")
        print("Check radgraph_debug.json for structure information")


if __name__ == '__main__':
    main()
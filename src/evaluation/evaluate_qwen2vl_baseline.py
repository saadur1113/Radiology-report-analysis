"""
Evaluate Qwen2-VL-7B as General-Purpose LLM Baseline

This provides a strong baseline comparison showing that:
- General-purpose LLMs can generate radiology reports
- But specialized models (yours) perform better

Usage:
    python evaluate_qwen2vl_baseline.py \
        --csv_file ./data/processed/mimic_cxr_val_clean.csv \
        --output_dir ./results/qwen2vl_baseline \
        --max_samples 1000
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import pandas as pd
from tqdm import tqdm
import json
import os
import argparse
from pathlib import Path


class Qwen2VLEvaluator:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct", device="cuda:0"):
        """Initialize Qwen2-VL model"""
        
        print("="*80)
        print("LOADING QWEN2-VL-7B (GENERAL-PURPOSE LLM BASELINE)")
        print("="*80)
        
        self.device = device
        
        print(f"\nLoading model: {model_name}")
        print("This may take a few minutes...")
        
        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        print(f"✓ Model loaded on {device}")
        print("="*80)
    
    def generate_report(self, image_path, max_new_tokens=150):
        """Generate radiology report from chest X-ray"""
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Create prompt
        prompt = """You are a radiologist. Analyze this chest X-ray and write a findings section for a radiology report. 
Be concise and focus on clinically relevant observations.

Findings:"""
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # Decode
        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Extract just the findings (after the prompt)
        if "Findings:" in generated_text:
            report = generated_text.split("Findings:")[-1].strip()
        else:
            report = generated_text.strip()
        
        return report
    
    def evaluate_dataset(self, csv_file, output_dir, max_samples=None):
        """Evaluate on MIMIC-CXR validation set"""
        
        print("\n" + "="*80)
        print("EVALUATING QWEN2-VL ON MIMIC-CXR")
        print("="*80)
        
        # Load data
        df = pd.read_csv(csv_file)
        if max_samples:
            df = df.head(max_samples)
        
        print(f"\nDataset: {len(df)} samples")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate predictions
        predictions = []
        references = []
        image_paths = []
        
        print("\nGenerating predictions...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
            try:
                image_path = row['image_path']
                reference = row['findings']
                
                # Generate
                prediction = self.generate_report(image_path)
                
                predictions.append(prediction)
                references.append(reference)
                image_paths.append(image_path)
                
            except Exception as e:
                print(f"\nError at index {idx}: {e}")
                predictions.append("")
                references.append(reference)
                image_paths.append(image_path)
                continue
        
        # Save predictions
        results = {
            'predictions': predictions,
            'references': references,
            'image_paths': image_paths
        }
        
        predictions_file = os.path.join(output_dir, 'predictions.json')
        with open(predictions_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Predictions saved: {predictions_file}")
        
        # Compute metrics
        print("\nComputing NLG metrics...")
        metrics = self.compute_metrics(predictions, references)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Metrics saved: {metrics_file}")
        
        # Print results
        print("\n" + "="*80)
        print("QWEN2-VL BASELINE RESULTS")
        print("="*80)
        print(f"\nBLEU-1:  {metrics['bleu_1']:.4f}")
        print(f"BLEU-4:  {metrics['bleu_4']:.4f}")
        print(f"ROUGE-L: {metrics['rouge_l']:.4f}")
        print(f"METEOR:  {metrics['meteor']:.4f}")
        print("="*80)
        
        return metrics
    
    def compute_metrics(self, predictions, references):
        """Compute BLEU, ROUGE, METEOR"""
        from evaluate import load
        
        # Load metrics
        bleu = load("bleu")
        rouge = load("rouge")
        meteor = load("meteor")
        
        # Compute
        bleu_scores = bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references],
            max_order=4
        )
        
        rouge_scores = rouge.compute(
            predictions=predictions,
            references=references
        )
        
        meteor_score = meteor.compute(
            predictions=predictions,
            references=references
        )
        
        return {
            'bleu_1': bleu_scores['precisions'][0],
            'bleu_2': bleu_scores['precisions'][1],
            'bleu_3': bleu_scores['precisions'][2],
            'bleu_4': bleu_scores['precisions'][3],
            'rouge_l': rouge_scores['rougeL'],
            'meteor': meteor_score['meteor']
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen2-VL Baseline')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Path to validation CSV')
    parser.add_argument('--output_dir', type=str, 
                       default='./results/qwen2vl_baseline',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit number of samples (None = all)')
    parser.add_argument('--model_name', type=str,
                       default='Qwen/Qwen2-VL-7B-Instruct',
                       help='Qwen2-VL model name')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Verify CSV exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    
    # Initialize evaluator
    evaluator = Qwen2VLEvaluator(
        model_name=args.model_name,
        device=args.device
    )
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )
    
    print("\n✓ Qwen2-VL baseline evaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

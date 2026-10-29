"""
Evaluate Qwen2-VL-7B with Prompt Engineering + One-Shot Example

This is the "best-case scenario" for a general-purpose LLM: we give Qwen2-VL
an expert radiologist persona, systematic instructions, and a worked example
in the exact MIMIC-CXR style. This isolates our model's architectural RAG
contribution from any prompt-weakness effect in the baseline.

Compare to:
  - evaluate_qwen2vl_baseline.py (zero-shot, minimal prompt)
  - Stage 1 / Stage 2 gen-then-retrieve (your fine-tuned models)

Usage:
    python evaluate_qwen2vl_prompt_eng.py \
        --csv_file ./data/processed/mimic_cxr_val_clean.csv \
        --output_dir ./results/qwen2vl_prompt_eng \
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


# =====================================================
# PROMPTS
# =====================================================
# Mirror the Gemini prompt_engineered mode for consistency

PROMPT_ENGINEERED = """You are an expert board-certified radiologist with 20 years of experience interpreting chest X-rays. Your task is to generate the FINDINGS section of a radiology report for the provided chest X-ray image.

## Instructions:
- Begin by identifying the view type (e.g., "PA and lateral views", "AP portable upright view", "Frontal view")
- Systematically describe the following in order:
  1. Heart size and mediastinal contours
  2. Lungs (each lobe / lung field) - consolidation, opacity, nodules, infiltrates
  3. Pleural spaces - effusion, pneumothorax
  4. Osseous structures - fractures, degenerative changes
  5. Support devices if present (tubes, lines, pacemakers)
- Note both normal AND abnormal findings
- Use precise standard radiology terminology (e.g., "focal consolidation", "pleural effusion", "cardiomediastinal silhouette")
- Be concise but thorough - typically 3-6 sentences
- Do NOT include impressions, recommendations, differential diagnoses, or clinical correlation
- Do NOT include patient information or indications
- Do NOT use phrases like "this chest X-ray shows" - write in the style of a radiologist's report

## Example of the target style and format:

Example output for a post-sternotomy chest X-ray with no acute findings:
"PA and lateral views of the chest were obtained. The patient is status post median sternotomy with intact sternal wires. The cardiomediastinal silhouette is within normal limits. The lungs are clear without focal consolidation, pleural effusion, or pneumothorax. No acute osseous abnormalities are identified."

## Now generate the FINDINGS section for the provided chest X-ray, following the exact style above. Output ONLY the findings text, no preamble or explanation:

FINDINGS:"""


class Qwen2VLPromptEngEvaluator:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct", device="cuda:0"):
        
        print("="*80)
        print("LOADING QWEN2-VL-7B (PROMPT-ENGINEERED BASELINE)")
        print("="*80)
        
        self.device = device
        
        print(f"\nLoading model: {model_name}")
        print("This may take a few minutes...")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        print(f"Model loaded on {device}")
        print("="*80)
    
    def generate_report(self, image_path, max_new_tokens=300):
        """Generate radiology report with prompt engineering"""
        
        image = Image.open(image_path).convert('RGB')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_ENGINEERED}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,  # greedy for speed; switch to beams=3 if quality too low
            )
        
        # Decode only the new tokens (skip the input prompt)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        
        generated_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Clean up - extract just the findings if the model echoes "FINDINGS:" or similar
        report = generated_text.strip()
        
        # Remove common prefixes the model might add
        prefixes_to_strip = [
            "FINDINGS:", "Findings:", "findings:",
            "Here is the findings section:",
            "Here's the findings section:",
            "The findings section:",
        ]
        for prefix in prefixes_to_strip:
            if report.startswith(prefix):
                report = report[len(prefix):].strip()
        
        # Remove quote marks if the whole response is quoted
        if report.startswith('"') and report.endswith('"'):
            report = report[1:-1].strip()
        
        return report
    
    def evaluate_dataset(self, csv_file, output_dir, max_samples=None, resume=False):
        """Evaluate on MIMIC-CXR validation set"""
        
        print("\n" + "="*80)
        print("EVALUATING QWEN2-VL (PROMPT ENGINEERED) ON MIMIC-CXR")
        print("="*80)
        
        df = pd.read_csv(csv_file)
        if max_samples:
            df = df.head(max_samples)
        
        print(f"\nDataset: {len(df)} samples")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Resume support
        progress_file = os.path.join(output_dir, 'progress.json')
        predictions = []
        references = []
        image_paths = []
        start_idx = 0
        
        if resume and os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            predictions = progress['predictions']
            references = progress['references']
            image_paths = progress['image_paths']
            start_idx = len(predictions)
            print(f"Resuming from sample {start_idx}")
        
        print(f"\nGenerating predictions ({start_idx} -> {len(df)})...")
        
        for idx in tqdm(range(start_idx, len(df)), desc="Qwen2-VL PE", initial=start_idx, total=len(df)):
            row = df.iloc[idx]
            try:
                image_path = row['image_path']
                reference = row['findings']
                
                prediction = self.generate_report(image_path)
                
                predictions.append(prediction)
                references.append(reference)
                image_paths.append(image_path)
                
            except Exception as e:
                print(f"\nError at index {idx}: {e}")
                predictions.append("")
                references.append(str(row.get('findings', '')))
                image_paths.append(str(row.get('image_path', '')))
                continue
            
            # Save progress every 50 samples
            if (idx + 1) % 50 == 0:
                with open(progress_file, 'w') as f:
                    json.dump({
                        'predictions': predictions,
                        'references': references,
                        'image_paths': image_paths,
                    }, f)
        
        # Save final predictions
        results = {
            'predictions': predictions,
            'references': references,
            'image_paths': image_paths,
            'prompt': PROMPT_ENGINEERED,
            'model': 'Qwen2-VL-7B-Instruct',
            'mode': 'prompt_engineered',
        }
        
        predictions_file = os.path.join(output_dir, 'predictions.json')
        with open(predictions_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nPredictions saved: {predictions_file}")
        
        # Clean up progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        # Metrics
        print("\nComputing NLG metrics...")
        metrics = self.compute_metrics(predictions, references)
        
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved: {metrics_file}")
        
        print("\n" + "="*80)
        print("QWEN2-VL PROMPT-ENGINEERED RESULTS")
        print("="*80)
        print(f"\nBLEU-1:  {metrics['bleu_1']:.4f}")
        print(f"BLEU-2:  {metrics['bleu_2']:.4f}")
        print(f"BLEU-3:  {metrics['bleu_3']:.4f}")
        print(f"BLEU-4:  {metrics['bleu_4']:.4f}")
        print(f"ROUGE-L: {metrics['rouge_l']:.4f}")
        print(f"METEOR:  {metrics['meteor']:.4f}")
        print("="*80)
        
        # Sample outputs for inspection
        print(f"\nSample outputs:")
        for i in range(min(3, len(predictions))):
            print(f"\n[{i}]")
            print(f"  Pred: {predictions[i][:200]}")
            print(f"  Ref:  {references[i][:200]}")
        
        print(f"\nNEXT STEP: Run RadGraph F1 evaluation:")
        print(f"  python evaluate_radgraph_final.py \\")
        print(f"    --predictions_file {predictions_file} \\")
        print(f"    --output_dir {output_dir}")
        
        return metrics
    
    def compute_metrics(self, predictions, references):
        """Compute BLEU, ROUGE, METEOR"""
        from evaluate import load
        
        bleu = load("bleu")
        rouge = load("rouge")
        meteor = load("meteor")
        
        # Filter empty predictions
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
        if not valid_pairs:
            return {'bleu_1': 0, 'bleu_2': 0, 'bleu_3': 0, 'bleu_4': 0, 'rouge_l': 0, 'meteor': 0}
        
        vp = [v[0] for v in valid_pairs]
        vr = [v[1] for v in valid_pairs]
        
        bleu_scores = bleu.compute(
            predictions=vp,
            references=[[ref] for ref in vr],
            max_order=4
        )
        
        rouge_scores = rouge.compute(predictions=vp, references=vr)
        meteor_score = meteor.compute(predictions=vp, references=vr)
        
        return {
            'bleu_1': bleu_scores['precisions'][0],
            'bleu_2': bleu_scores['precisions'][1],
            'bleu_3': bleu_scores['precisions'][2],
            'bleu_4': bleu_scores['precisions'][3],
            'rouge_l': rouge_scores['rougeL'],
            'meteor': meteor_score['meteor'],
            'n_valid': len(valid_pairs),
            'n_total': len(predictions),
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen2-VL with Prompt Engineering')
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results/qwen2vl_prompt_eng')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    
    evaluator = Qwen2VLPromptEngEvaluator(
        model_name=args.model_name,
        device=args.device
    )
    
    evaluator.evaluate_dataset(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        resume=args.resume,
    )
    
    print("\nQwen2-VL prompt-engineered evaluation complete!")


if __name__ == '__main__':
    main()

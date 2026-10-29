"""
Comprehensive Evaluation Script for RAG-Enhanced BLIP2 - FAST VERSION
BERTScore SKIPPED (saves ~4 hours per evaluation)

Features:
- Evaluates both Stage 1 (baseline) and Stage 2 (RAG) models
- Handles corrupted images gracefully
- GPU memory management
- Progress tracking and resumability
- Fast metrics: BLEU, ROUGE, METEOR (BERTScore skipped)
- Error logging
- Automatic result saving
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from pathlib import Path
import pickle

# Add preprocessing to path

from modified_qformer_opt import RAGEnhancedBLIP2_OPT
from dataset_opt import MIMICCXRDataset, MIMICCXRDatasetWithRetrieval, collate_fn
from torch.utils.data import DataLoader


class ComprehensiveEvaluator:
    """
    Robust evaluator with error handling and progress tracking
    """
    
    def __init__(
        self,
        checkpoint_path,
        output_dir,
        use_retrieval=False,
        retrieval_dir='./data/retrieval',
        batch_size=8,
        max_samples=None,
        device='cuda:0'
    ):
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_retrieval = use_retrieval
        self.retrieval_dir = retrieval_dir
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Error tracking
        self.errors = []
        self.skipped_samples = []
        
        # Results storage
        self.predictions = []
        self.references = []
        self.sample_ids = []
        
        print("="*80)
        print("COMPREHENSIVE EVALUATION INITIALIZATION (FAST MODE)")
        print("="*80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Output dir: {output_dir}")
        print(f"Use retrieval: {use_retrieval}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.device}")
        print(f"⚡ BERTScore: SKIPPED (saves ~4 hours)")
        print("="*80)
    
    def load_model(self):
        """Load model with error handling"""
        print("\n[1/5] Loading model...")
        
        try:
            # Initialize model
            self.model = RAGEnhancedBLIP2_OPT(
                blip2_model_name='Salesforce/blip2-opt-6.7b',
                num_query_tokens=32,
                freeze_vision=True,
                freeze_llm=True,
                use_retrieval=self.use_retrieval
            )
            
            # Load checkpoint
            print(f"Loading checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                val_loss = checkpoint.get('val_loss', 'unknown')
                print(f"Checkpoint info: Epoch {epoch}, Val Loss {val_loss}")
            else:
                state_dict = checkpoint
            
            # Load state dict
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"✓ Model loaded successfully")
            if missing:
                print(f"  Missing keys: {len(missing)} (expected if switching retrieval on/off)")
            
            # Move to device and eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✓ Model on {self.device} in eval mode")
            
            return True
            
        except Exception as e:
            print(f"✗ ERROR loading model: {e}")
            self.errors.append(f"Model loading error: {e}")
            return False
    
    def load_dataset(self, csv_file):
        """Load dataset with error handling"""
        print(f"\n[2/5] Loading dataset from {csv_file}...")
        
        try:
            if self.use_retrieval:
                print("Using dataset WITH retrieval")
                dataset = MIMICCXRDatasetWithRetrieval(
                    csv_file=csv_file,
                    retrieval_dir=self.retrieval_dir,
                    tokenizer_name='Salesforce/blip2-opt-6.7b',
                    max_length=128,
                    k=3
                )
            else:
                print("Using dataset WITHOUT retrieval")
                dataset = MIMICCXRDataset(
                    csv_file=csv_file,
                    tokenizer_name='Salesforce/blip2-opt-6.7b',
                    max_length=128
                )
            
            # Limit samples if requested
            if self.max_samples and self.max_samples < len(dataset):
                print(f"Limiting to {self.max_samples} samples (out of {len(dataset)})")
                # Use a subset
                from torch.utils.data import Subset
                indices = list(range(self.max_samples))
                dataset = Subset(dataset, indices)
            
            print(f"✓ Dataset loaded: {len(dataset)} samples")
            
            # Create DataLoader
            # CRITICAL: Use num_workers=0 to avoid CUDA multiprocessing issues
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,  # ← IMPORTANT!
                collate_fn=collate_fn,
                pin_memory=True
            )
            
            print(f"✓ DataLoader created: {len(dataloader)} batches")
            
            self.dataset = dataset
            self.dataloader = dataloader
            return True
            
        except Exception as e:
            print(f"✗ ERROR loading dataset: {e}")
            self.errors.append(f"Dataset loading error: {e}")
            return False
    
    def generate_reports(self):
        """Generate reports with comprehensive error handling"""
        print(f"\n[3/5] Generating reports...")
        print(f"Total batches: {len(self.dataloader)}")
        print(f"Batch size: {self.batch_size}")
        
        # Resume capability
        progress_file = self.output_dir / 'generation_progress.json'
        if progress_file.exists():
            print(f"Found existing progress file. Loading...")
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                self.predictions = progress.get('predictions', [])
                self.references = progress.get('references', [])
                self.sample_ids = progress.get('sample_ids', [])
                start_batch = len(self.predictions) // self.batch_size
                print(f"Resuming from batch {start_batch}")
        else:
            start_batch = 0
        
        # Generation parameters
        generation_config = {
            'max_length': 100,
            'num_beams': 5,
            'early_stopping': True,
            'no_repeat_ngram_size': 3,
            'length_penalty': 1.0,
        }
        
        print(f"Generation config: {generation_config}")
        
        # Progress tracking
        successful = 0
        failed = 0
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(
                    self.dataloader, 
                    desc="Generating",
                    initial=start_batch
                )):
                    # Skip if already processed
                    if batch_idx < start_batch:
                        continue
                    
                    try:
                        # Move to device
                        pixel_values = batch['pixel_values'].to(self.device)
                        retrieved_texts = batch.get('retrieved_texts', None)
                        
                        # Generate
                        outputs = self.model.generate(
                            pixel_values=pixel_values,
                            retrieved_texts=retrieved_texts,
                            **generation_config
                        )
                        
                        # Decode predictions
                        tokenizer = self.model.tokenizer
                        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        
                        # Get references (ground truth)
                        references = batch.get('text', [''] * len(predictions))
                        
                        # Store results
                        for pred, ref in zip(predictions, references):
                            self.predictions.append(pred)
                            self.references.append(ref)
                            self.sample_ids.append(successful)
                            successful += 1
                        
                        # Save progress every 100 batches
                        if (batch_idx + 1) % 100 == 0:
                            self._save_progress()
                        
                    except Exception as e:
                        print(f"\nError in batch {batch_idx}: {e}")
                        self.errors.append(f"Batch {batch_idx} error: {e}")
                        failed += 1
                        
                        # Add dummy predictions to maintain alignment
                        batch_size_actual = len(batch['pixel_values'])
                        for _ in range(batch_size_actual):
                            self.predictions.append("")
                            self.references.append("")
                            self.sample_ids.append(-1)  # Mark as failed
                        
                        continue
                    
                    # GPU memory cleanup
                    if (batch_idx + 1) % 50 == 0:
                        torch.cuda.empty_cache()
            
            print(f"\n✓ Generation complete!")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            
            # Final save
            self._save_progress()
            self._save_raw_outputs()
            
            return True
            
        except Exception as e:
            print(f"\n✗ CRITICAL ERROR during generation: {e}")
            self.errors.append(f"Generation critical error: {e}")
            # Save what we have
            self._save_progress()
            self._save_raw_outputs()
            return False
    
    def calculate_metrics(self):
        """Calculate all evaluation metrics"""
        print(f"\n[4/5] Calculating metrics...")
        
        # Filter out failed samples
        valid_indices = [i for i, sid in enumerate(self.sample_ids) if sid >= 0]
        valid_predictions = [self.predictions[i] for i in valid_indices]
        valid_references = [self.references[i] for i in valid_indices]
        
        print(f"Valid samples: {len(valid_predictions)}")
        print(f"Failed samples: {len(self.predictions) - len(valid_predictions)}")
        
        if len(valid_predictions) == 0:
            print("✗ No valid predictions to evaluate!")
            return False
        
        metrics = {}
        
        # 1. BLEU Scores
        print("\nCalculating BLEU scores...")
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize
            import nltk
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            # Tokenize
            references_tokenized = [[word_tokenize(ref.lower())] for ref in valid_references]
            predictions_tokenized = [word_tokenize(pred.lower()) for pred in valid_predictions]
            
            smoothing = SmoothingFunction().method1
            
            bleu1 = corpus_bleu(references_tokenized, predictions_tokenized, 
                               weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu2 = corpus_bleu(references_tokenized, predictions_tokenized,
                               weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu3 = corpus_bleu(references_tokenized, predictions_tokenized,
                               weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu4 = corpus_bleu(references_tokenized, predictions_tokenized,
                               weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            metrics['bleu1'] = bleu1
            metrics['bleu2'] = bleu2
            metrics['bleu3'] = bleu3
            metrics['bleu4'] = bleu4
            
            print(f"✓ BLEU-1: {bleu1:.4f}")
            print(f"✓ BLEU-2: {bleu2:.4f}")
            print(f"✓ BLEU-3: {bleu3:.4f}")
            print(f"✓ BLEU-4: {bleu4:.4f}")
            
        except Exception as e:
            print(f"✗ BLEU calculation failed: {e}")
            metrics['bleu1'] = metrics['bleu2'] = metrics['bleu3'] = metrics['bleu4'] = None
        
        # 2. ROUGE-L
        print("\nCalculating ROUGE-L...")
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_scores = []
            
            for pred, ref in zip(valid_predictions, valid_references):
                score = scorer.score(ref, pred)
                rouge_scores.append(score['rougeL'].fmeasure)
            
            rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
            metrics['rouge_l'] = rouge_l
            
            print(f"✓ ROUGE-L: {rouge_l:.4f}")
            
        except Exception as e:
            print(f"✗ ROUGE calculation failed: {e}")
            metrics['rouge_l'] = None
        
        # 3. METEOR
        print("\nCalculating METEOR...")
        try:
            from nltk.translate.meteor_score import meteor_score
            import nltk
            
            # Download required data
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            
            meteor_scores = []
            for pred, ref in zip(valid_predictions, valid_references):
                score = meteor_score([word_tokenize(ref.lower())], 
                                    word_tokenize(pred.lower()))
                meteor_scores.append(score)
            
            meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
            metrics['meteor'] = meteor
            
            print(f"✓ METEOR: {meteor:.4f}")
            
        except Exception as e:
            print(f"✗ METEOR calculation failed: {e}")
            metrics['meteor'] = None
        
        # 4. BERTScore - SKIPPED (saves ~4 hours)
        print("\n⚠️  Skipping BERTScore (saves ~4 hours)")
        print("   BLEU, ROUGE, and METEOR are sufficient for thesis")
        metrics['bertscore_f1'] = None
        metrics['bertscore_precision'] = None
        metrics['bertscore_recall'] = None
        
        # Save metrics
        self.metrics = metrics
        self._save_metrics()
        
        return True
    
    def _save_progress(self):
        """Save generation progress for resumability"""
        progress_file = self.output_dir / 'generation_progress.json'
        progress = {
            'predictions': self.predictions,
            'references': self.references,
            'sample_ids': self.sample_ids,
            'timestamp': datetime.now().isoformat()
        }
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def _save_raw_outputs(self):
        """Save raw predictions and references"""
        print("\nSaving raw outputs...")
        
        # Save as JSON
        outputs = {
            'predictions': self.predictions,
            'references': self.references,
            'sample_ids': self.sample_ids,
            'use_retrieval': self.use_retrieval,
            'checkpoint': str(self.checkpoint_path),
            'timestamp': datetime.now().isoformat()
        }
        
        output_file = self.output_dir / 'predictions.json'
        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=2)
        print(f"✓ Saved to {output_file}")
        
        # Also save as CSV for easy viewing
        df = pd.DataFrame({
            'sample_id': self.sample_ids,
            'prediction': self.predictions,
            'reference': self.references
        })
        csv_file = self.output_dir / 'predictions.csv'
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved to {csv_file}")
    
    def _save_metrics(self):
        """Save calculated metrics"""
        print("\nSaving metrics...")
        
        metrics_file = self.output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✓ Saved to {metrics_file}")
        
        # Also save as readable text
        report_file = self.output_dir / 'metrics_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EVALUATION METRICS REPORT (FAST MODE)\n")
            f.write("="*80 + "\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Use Retrieval: {self.use_retrieval}\n")
            f.write(f"Samples Evaluated: {len([s for s in self.sample_ids if s >= 0])}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Note: BERTScore skipped (saves ~4 hours)\n")
            f.write("\n" + "="*80 + "\n")
            f.write("METRICS:\n")
            f.write("="*80 + "\n")
            
            for metric_name, value in self.metrics.items():
                if value is not None:
                    f.write(f"{metric_name.upper():20s}: {value:.4f}\n")
                else:
                    f.write(f"{metric_name.upper():20s}: N/A (skipped)\n")
            
            f.write("="*80 + "\n")
        
        print(f"✓ Saved to {report_file}")
    
    def save_error_log(self):
        """Save error log"""
        if self.errors:
            print(f"\n[5/5] Saving error log ({len(self.errors)} errors)...")
            error_file = self.output_dir / 'errors.log'
            with open(error_file, 'w') as f:
                f.write(f"ERROR LOG - {datetime.now().isoformat()}\n")
                f.write("="*80 + "\n\n")
                for i, error in enumerate(self.errors, 1):
                    f.write(f"Error {i}:\n{error}\n\n")
            print(f"✓ Error log saved to {error_file}")
        else:
            print("\n[5/5] No errors to log ✓")
    
    def run_evaluation(self, csv_file):
        """Run complete evaluation pipeline"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE EVALUATION (FAST MODE)")
        print("="*80)
        
        start_time = datetime.now()
        
        # Step 1: Load model
        if not self.load_model():
            print("\n✗ Evaluation failed at model loading")
            self.save_error_log()
            return False
        
        # Step 2: Load dataset
        if not self.load_dataset(csv_file):
            print("\n✗ Evaluation failed at dataset loading")
            self.save_error_log()
            return False
        
        # Step 3: Generate reports
        if not self.generate_reports():
            print("\n✗ Evaluation failed at generation (partial results saved)")
            self.save_error_log()
            return False
        
        # Step 4: Calculate metrics
        if not self.calculate_metrics():
            print("\n✗ Evaluation failed at metrics calculation")
            self.save_error_log()
            return False
        
        # Step 5: Save error log
        self.save_error_log()
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        print(f"Duration: {duration}")
        print(f"Output directory: {self.output_dir}")
        print(f"Samples evaluated: {len([s for s in self.sample_ids if s >= 0])}")
        print(f"Errors encountered: {len(self.errors)}")
        print(f"⚡ Time saved by skipping BERTScore: ~4 hours")
        print("\nResults saved:")
        print(f"  - predictions.json/csv")
        print(f"  - metrics.json")
        print(f"  - metrics_report.txt")
        if self.errors:
            print(f"  - errors.log")
        print("="*80)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation (Fast)')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    
    # Dataset arguments
    parser.add_argument('--csv_file', type=str,
                       default='./data/processed/mimic_cxr_val_clean.csv',
                       help='Path to validation CSV')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to evaluate (None = all)')
    
    # Model arguments
    parser.add_argument('--use_retrieval', action='store_true',
                       help='Use retrieval (for Stage 2 model)')
    parser.add_argument('--retrieval_dir', type=str,
                       default='./data/retrieval',
                       help='Retrieval database directory')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"✗ ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    # Verify CSV exists
    if not os.path.exists(args.csv_file):
        print(f"✗ ERROR: CSV file not found: {args.csv_file}")
        return
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        use_retrieval=args.use_retrieval,
        retrieval_dir=args.retrieval_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device
    )
    
    # Run evaluation
    success = evaluator.run_evaluation(args.csv_file)
    
    if success:
        print("\n✅ Evaluation completed successfully!")
        return 0
    else:
        print("\n⚠️ Evaluation completed with errors (partial results saved)")
        return 1


if __name__ == '__main__':
    exit(main())
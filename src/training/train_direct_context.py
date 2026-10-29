"""
Direct Context Training: Fine-tune OPT to accept retrieved 
reports directly in its input alongside Q-Former visual embeddings.

Tests whether retrieved reports contain useful information 
when properly delivered to the decoder.

Pipeline:
  [retrieved text tokens] + [Q-Former visual embeddings] → OPT → report

Two conditions evaluated after training:
  1. Visual only (no context) — baseline
  2. Visual + direct context — does retrieval help?

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_direct_context.py \
        --stage1_checkpoint ./checkpoints/stage1_retrained/best_model.pt \
        --train_csv ./data/processed/mimic_cxr_train_clean.csv \
        --val_csv ./data/processed/mimic_cxr_val_clean.csv \
        --retrieval_cache_train ./data/retrained_gtr/retrieval_cache_train_gtr.pkl \
        --retrieval_cache_val ./data/retrained_gtr/retrieval_cache_val_gtr.pkl \
        --output_dir ./checkpoints/direct_context \
        --epochs 5 --batch_size 4 --device cuda:0
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import torch
import torch.nn as nn
import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random

import sys
from modified_qformer_opt import RAGEnhancedBLIP2_OPT


class DirectContextDataset(Dataset):
    """Dataset that provides retrieved text for direct LLM injection."""
    
    def __init__(self, csv_file, tokenizer, retrieval_cache_path, 
                 max_length=128, max_context_length=200):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_context_length = max_context_length
        
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        with open(retrieval_cache_path, 'rb') as f:
            self.cache = pickle.load(f)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        print(f"DirectContextDataset: {len(self.df)} samples, "
              f"cache: {len(self.cache)} entries")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        try:
            image = Image.open(row['image_path']).convert('RGB')
            image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)
        
        # GT findings
        findings = str(row.get('findings', ''))
        
        # Retrieved context
        retrieved = self.cache.get(idx, [''] * 3)
        context_text = "Context: " + " ".join(retrieved) + " Findings: "
        
        # Tokenize context
        context_encoded = self.tokenizer(
            context_text,
            max_length=self.max_context_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_text = findings + self.tokenizer.eos_token
        target_encoded = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = target_encoded['input_ids'].squeeze(0)
        attention_mask = target_encoded['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'pixel_values': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'context_input_ids': context_encoded['input_ids'].squeeze(0),
            'context_attention_mask': context_encoded['attention_mask'].squeeze(0),
            'text': findings,
        }


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch]),
        'context_input_ids': torch.stack([x['context_input_ids'] for x in batch]),
        'context_attention_mask': torch.stack([x['context_attention_mask'] for x in batch]),
    }


class DirectContextModel(nn.Module):
    """Wraps the BLIP2 model to inject text context directly into OPT."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.vision_model = base_model.vision_model
        self.qformer = base_model.qformer
        self.language_projection = base_model.language_projection
        self.language_model = base_model.language_model
        self.tokenizer = base_model.tokenizer
        
        # Freeze vision + Q-Former, unfreeze LLM
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.qformer.parameters():
            param.requires_grad = False
        for param in self.language_projection.parameters():
            param.requires_grad = False
        
        # Unfreeze LLM with lower learning rate
        for param in self.language_model.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")
    
    def forward(self, pixel_values, context_input_ids, context_attention_mask,
                input_ids, attention_mask, labels):
        
        with torch.no_grad():
            # Get visual embeddings through Q-Former
            vision_outputs = self.vision_model(pixel_values, return_dict=True)
            image_embeds = vision_outputs.last_hidden_state
            
            # Project vision features 1408 -> 768
            image_embeds = self.base_model.vision_projection(image_embeds)
            
            query_outputs = self.qformer(
                encoder_hidden_states=image_embeds,
                retrieval_features=None,
                retrieval_mask=None,
                return_dict=True
            )
            
            if hasattr(query_outputs, 'last_hidden_state'):
                qf_embeds = query_outputs.last_hidden_state
            elif isinstance(query_outputs, tuple):
                qf_embeds = query_outputs[0]
            else:
                qf_embeds = query_outputs
            
            visual_embeds = self.language_projection(qf_embeds)
        
        # Get context token embeddings
        context_embeds = self.language_model.get_input_embeddings()(context_input_ids)
        
        # Get target token embeddings
        target_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # Combine: [context_text] + [visual] + [target]
        batch_size = pixel_values.shape[0]
        visual_attn = torch.ones(
            batch_size, visual_embeds.shape[1],
            dtype=torch.long, device=pixel_values.device
        )
        
        combined_embeds = torch.cat([
            context_embeds, visual_embeds, target_embeds
        ], dim=1)
        
        combined_attn = torch.cat([
            context_attention_mask, visual_attn, attention_mask
        ], dim=1)
        
        # Labels: -100 for context and visual positions, real labels for target
        context_labels = torch.full(
            (batch_size, context_embeds.shape[1] + visual_embeds.shape[1]),
            -100, dtype=torch.long, device=pixel_values.device
        )
        combined_labels = torch.cat([context_labels, labels], dim=1)
        
        # Forward through LLM
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attn,
            labels=combined_labels,
            return_dict=True,
        )
        
        return outputs
    
    def generate_with_context(self, pixel_values, context_input_ids, 
                               context_attention_mask, **gen_kwargs):
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values, return_dict=True)
            image_embeds = vision_outputs.last_hidden_state
            
            image_embeds = self.base_model.vision_projection(image_embeds)
            
            query_outputs = self.qformer(
                encoder_hidden_states=image_embeds,
                retrieval_features=None,
                retrieval_mask=None,
                return_dict=True
            )
            
            if hasattr(query_outputs, 'last_hidden_state'):
                qf_embeds = query_outputs.last_hidden_state
            elif isinstance(query_outputs, tuple):
                qf_embeds = query_outputs[0]
            else:
                qf_embeds = query_outputs
            
            visual_embeds = self.language_projection(qf_embeds)
            
            context_embeds = self.language_model.get_input_embeddings()(context_input_ids)
            
            combined = torch.cat([context_embeds, visual_embeds], dim=1)
            visual_attn = torch.ones(
                visual_embeds.shape[:2], dtype=torch.long, 
                device=pixel_values.device
            )
            combined_attn = torch.cat([
                context_attention_mask, visual_attn
            ], dim=1)
            
            outputs = self.language_model.generate(
                inputs_embeds=combined,
                attention_mask=combined_attn,
                **gen_kwargs
            )
            
            return outputs
    
    def generate_without_context(self, pixel_values, **gen_kwargs):
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values, return_dict=True)
            image_embeds = vision_outputs.last_hidden_state
            
            image_embeds = self.base_model.vision_projection(image_embeds)
            
            query_outputs = self.qformer(
                encoder_hidden_states=image_embeds,
                retrieval_features=None,
                retrieval_mask=None,
                return_dict=True
            )
            
            if hasattr(query_outputs, 'last_hidden_state'):
                qf_embeds = query_outputs.last_hidden_state
            elif isinstance(query_outputs, tuple):
                qf_embeds = query_outputs[0]
            else:
                qf_embeds = query_outputs
            
            visual_embeds = self.language_projection(qf_embeds)
            visual_attn = torch.ones(
                visual_embeds.shape[:2], dtype=torch.long,
                device=pixel_values.device
            )
            
            outputs = self.language_model.generate(
                inputs_embeds=visual_embeds,
                attention_mask=visual_attn,
                **gen_kwargs
            )
            
            return outputs


def evaluate(model, df, cache, transform, device, n=500):
    """Evaluate both with and without context."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    for pkg, name in [('tokenizers/punkt', 'punkt'), 
                      ('tokenizers/punkt_tab', 'punkt_tab')]:
        try: nltk.data.find(pkg)
        except: nltk.download(name, quiet=True)
    
    preds_no = []
    preds_with = []
    refs = []
    
    for idx in tqdm(range(min(n, len(df))), desc="Evaluating"):
        try:
            row = df.iloc[idx]
            image = Image.open(row['image_path']).convert('RGB')
            pixel_values = transform(image).unsqueeze(0).to(device)
            retrieved = cache.get(idx, [''] * 3)
            gt = str(row['findings'])
            refs.append(gt)
            
            gen_kwargs = dict(max_new_tokens=100, num_beams=5,
                            early_stopping=True, no_repeat_ngram_size=3,
                            length_penalty=1.0)
            
            # Without context
            out_no = model.generate_without_context(pixel_values, **gen_kwargs)
            gen_no = model.tokenizer.decode(out_no[0], skip_special_tokens=True)
            preds_no.append(gen_no)
            
            # With context
            context = "Context: " + " ".join(retrieved) + " Findings: "
            ctx_tok = model.tokenizer(
                context, return_tensors='pt',
                truncation=True, max_length=200, padding=True
            ).to(device)
            
            out_with = model.generate_with_context(
                pixel_values, ctx_tok.input_ids, 
                ctx_tok.attention_mask, **gen_kwargs
            )
            gen_with = model.tokenizer.decode(out_with[0], skip_special_tokens=True)
            preds_with.append(gen_with)
            
        except Exception as e:
            preds_no.append("")
            preds_with.append("")
            refs.append("")
    
    smooth = SmoothingFunction().method1
    
    def bleu4(preds, references):
        valid = [(p, r) for p, r in zip(preds, references) 
                 if p.strip() and r.strip()]
        if not valid: return 0, 0
        vp, vr = zip(*valid)
        refs_tok = [[word_tokenize(r.lower())] for r in vr]
        preds_tok = [word_tokenize(p.lower()) for p in vp]
        b4 = corpus_bleu(refs_tok, preds_tok, 
                         weights=(0.25,0.25,0.25,0.25),
                         smoothing_function=smooth)
        avg_len = np.mean([len(p.split()) for p in vp])
        return b4, avg_len
    
    b4_no, len_no = bleu4(preds_no, refs)
    b4_with, len_with = bleu4(preds_with, refs)
    
    return {
        'no_context': {'bleu4': b4_no, 'avg_words': len_no},
        'with_context': {'bleu4': b4_with, 'avg_words': len_with},
        'delta': b4_with - b4_no,
        'preds_no': preds_no,
        'preds_with': preds_with,
        'refs': refs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_checkpoint', required=True)
    parser.add_argument('--train_csv', default='./data/processed/mimic_cxr_train_clean.csv')
    parser.add_argument('--val_csv', default='./data/processed/mimic_cxr_val_clean.csv')
    parser.add_argument('--retrieval_cache_train', required=True)
    parser.add_argument('--retrieval_cache_val', required=True)
    parser.add_argument('--output_dir', default='./checkpoints/direct_context')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--eval_samples', type=int, default=500)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print("=" * 70)
    print("DIRECT CONTEXT TRAINING")
    print("Fine-tune OPT to use retrieved reports directly")
    print("=" * 70)
    
    # Load base model
    print("\nLoading model...")
    base_model = RAGEnhancedBLIP2_OPT(
        blip2_model_name='Salesforce/blip2-opt-2.7b',
        num_query_tokens=32,
        freeze_vision=True,
        freeze_llm=True,
        use_retrieval=False
    )
    ckpt = torch.load(args.stage1_checkpoint, map_location='cpu')
    base_model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    
    # Wrap with direct context model
    model = DirectContextModel(base_model).to(device)
    
    # Dataset
    print("\nLoading data...")
    train_dataset = DirectContextDataset(
        args.train_csv, model.tokenizer, args.retrieval_cache_train
    )
    val_dataset = DirectContextDataset(
        args.val_csv, model.tokenizer, args.retrieval_cache_val
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    
    # Only optimize LLM parameters with low LR
    optimizer = torch.optim.AdamW(
        [p for p in model.language_model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=0.01
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )
    
    best_val_loss = float('inf')
    
    print(f"\nTraining for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            try:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                context_ids = batch['context_input_ids'].to(device)
                context_mask = batch['context_attention_mask'].to(device)
                
                outputs = model(
                    pixel_values=pixel_values,
                    context_input_ids=context_ids,
                    context_attention_mask=context_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    avg = train_loss / (batch_idx + 1)
                    print(f"  Step {batch_idx}: loss={avg:.4f}")
                    
            except Exception as e:
                print(f"  Error batch {batch_idx}: {e}")
                optimizer.zero_grad()
                continue
        
        avg_train = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    outputs = model(
                        pixel_values=batch['pixel_values'].to(device),
                        context_input_ids=batch['context_input_ids'].to(device),
                        context_attention_mask=batch['context_attention_mask'].to(device),
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device)
                    )
                    val_loss += outputs.loss.item()
                except:
                    continue
        
        avg_val = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  Saved best model (val_loss={avg_val:.4f})")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    val_df = pd.read_csv(args.val_csv)
    val_cache = pickle.load(open(args.retrieval_cache_val, 'rb'))
    
    results = evaluate(model, val_df, val_cache, transform, device, 
                       n=args.eval_samples)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  No context:   BLEU-4={results['no_context']['bleu4']:.4f} "
          f"({results['no_context']['avg_words']:.1f}w)")
    print(f"  With context: BLEU-4={results['with_context']['bleu4']:.4f} "
          f"({results['with_context']['avg_words']:.1f}w)")
    print(f"  Delta:        {results['delta']:+.4f}")
    print(f"{'='*70}")
    
    if results['delta'] > 0.005:
        print("  VERDICT: Direct context HELPS")
        print("  Retrieval IS useful — Q-Former is the bottleneck")
    elif results['delta'] > 0:
        print("  VERDICT: Small positive effect")
    else:
        print("  VERDICT: Direct context does not help")
    
    # Save
    save_results = {
        'no_context': results['no_context'],
        'with_context': results['with_context'],
        'delta': results['delta'],
        'best_val_loss': best_val_loss,
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)
    
    with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
        json.dump({
            'no_context': results['preds_no'],
            'with_context': results['preds_with'],
            'references': results['refs'],
        }, f, indent=2)
    
    print(f"\nSaved to {args.output_dir}/")


if __name__ == '__main__':
    main()

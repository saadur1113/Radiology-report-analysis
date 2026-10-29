"""
Dataset for OPT-based BLIP2 Training

Key differences from T5 version:
1. OPT uses GPT-2 tokenizer (different from T5)
2. Left padding for generation (vs right padding)
3. Different special tokens

This handles all OPT-specific tokenization correctly!
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import pickle
from transformers import AutoTokenizer


class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR Dataset for OPT-based training
    
    Handles:
    - Image loading and preprocessing
    - Text tokenization for OPT
    - Proper padding and truncation
    """
    
    def __init__(
        self,
        csv_file: str,
        tokenizer_name: str = "Salesforce/blip2-opt-2.7b",
        max_length: int = 128,
        image_size: int = 224,
    ):
        """
        Args:
            csv_file: Path to CSV with columns: image_path, findings
            tokenizer_name: OPT tokenizer name
            max_length: Maximum sequence length
            image_size: Image size for resizing
        """
        self.df = pd.read_csv(csv_file)
        self.max_length = max_length
        
        # Load OPT tokenizer
        print(f"Loading OPT tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # CRITICAL: Set padding side for OPT
        # OPT needs left padding for generation
        self.tokenizer.padding_side = 'right'  # For training, use right padding
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✓ Tokenizer loaded")
        print(f"  Vocab size: {len(self.tokenizer)}")
        print(f"  Pad token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        print(f"  EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        print(f"✓ Dataset initialized with {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        try:
            image = Image.open(row['image_path']).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {row['image_path']}: {e}")
            # Return dummy image on error
            image = torch.zeros(3, 224, 224)
        
        # Get text
        findings = str(row.get('findings', ''))
        
        # Tokenize for OPT
        # Add EOS token at the end for proper generation
        text_with_eos = findings + self.tokenizer.eos_token
        
        encoded = self.tokenizer(
            text_with_eos,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Create labels (same as input_ids, but mask padding)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss
        
        return {
            'pixel_values': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': findings  # For debugging
        }


class RetrievalSystem:
    """
    Retrieval system for fetching similar reports
    Uses FAISS for efficient similarity search
    """
    
    def __init__(self, retrieval_dir: str):
        """
        Args:
            retrieval_dir: Directory containing:
                - retrieval_metadata.csv (with text column)
                - report_index.faiss (FAISS index)
                - config.json (encoder config)
        """
        self.retrieval_dir = retrieval_dir
        
        print(f"Loading retrieval system from {retrieval_dir}...")
        
        # Load config
        import json
        config_path = os.path.join(retrieval_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            encoder_name = config.get('encoder_name', 'all-MiniLM-L6-v2')
            text_column = config.get('text_column', 'combined_text')
        else:
            encoder_name = 'all-MiniLM-L6-v2'
            text_column = 'combined_text'
        
        # Load report database from CSV
        import pandas as pd
        metadata_path = os.path.join(retrieval_dir, 'retrieval_metadata.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
        
        metadata = pd.read_csv(metadata_path)
        
        # Extract report texts
        if text_column not in metadata.columns:
            raise ValueError(f"Column '{text_column}' not found in metadata. Available: {metadata.columns.tolist()}")
        
        self.report_database = metadata[text_column].fillna('').tolist()
        
        # Load FAISS index
        import faiss
        index_path = os.path.join(retrieval_dir, 'report_index.faiss')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load encoder on GPU (safe with num_workers=0!)
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Use GPU if available (no multiprocessing issues with num_workers=0)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"  Loading SentenceTransformer encoder on {self.device}...")
        self.encoder = SentenceTransformer(encoder_name)
        self.encoder = self.encoder.to(self.device)
        
        # Warm up the encoder
        print(f"  Warming up encoder...")
        with torch.no_grad():
            _ = self.encoder.encode(
                ["warmup test"],
                batch_size=1,
                show_progress_bar=True,
                device=self.device
            )
        
        print(f"✓ Retrieval system loaded")
        print(f"  Database size: {len(self.report_database)} reports")
        print(f"  Index size: {self.index.ntotal} vectors")
        print(f"  Encoder: {encoder_name}")
        print(f"  Device: {self.device} ← GPU = FAST!")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve k most similar reports
        
        Args:
            query: Query text (e.g., ground truth report)
            k: Number of reports to retrieve
        
        Returns:
            List of dicts with 'text' and 'similarity' keys
        """
        import numpy as np
        import faiss
        import torch
        
        # Encode query on GPU (FAST!)
        try:
            with torch.no_grad():
                query_embedding = self.encoder.encode(
                    [query],
                    batch_size=1,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=self.device  # Use GPU!
                )
        except Exception as e:
            print(f"Warning: Encoding failed: {e}")
            return [{'text': '', 'similarity': 0.0} for _ in range(k)]
        
        # Ensure 2D array and normalize
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        try:
            similarities, indices = self.index.search(
                query_embedding.astype('float32'), k
            )
        except Exception as e:
            print(f"Warning: FAISS search failed: {e}")
            return [{'text': '', 'similarity': 0.0} for _ in range(k)]
        
        # Get reports
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx >= 0 and idx < len(self.report_database):
                results.append({
                    'text': self.report_database[idx],
                    'similarity': float(sim)
                })
        
        # Pad if needed
        while len(results) < k:
            results.append({'text': '', 'similarity': 0.0})
        
        return results[:k]


class MIMICCXRDatasetWithRetrieval(MIMICCXRDataset):
    """
    Dataset with retrieval augmentation for Stage 2
    """
    
    def __init__(
        self,
        csv_file: str,
        retrieval_dir: str,
        tokenizer_name: str = "Salesforce/blip2-opt-2.7b",
        max_length: int = 128,
        image_size: int = 224,
        k: int = 3,
    ):
        super().__init__(csv_file, tokenizer_name, max_length, image_size)
        
        self.k = k
        self.retrieval_system = RetrievalSystem(retrieval_dir)
        
        print(f"✓ Retrieval enabled (k={k})")
    
    def __getitem__(self, idx):
        # Get base item
        item = super().__getitem__(idx)
        
        # Retrieve similar reports
        findings = item['text']
        retrieved = self.retrieval_system.retrieve(findings, k=self.k)
        
        # Extract texts
        retrieved_texts = [r['text'] for r in retrieved]
        
        # Add to item
        item['retrieved_texts'] = retrieved_texts
        
        return item


def collate_fn(batch):
    """
    Custom collate function for batching
    Handles variable-length retrieved texts
    """
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    texts = [item['text'] for item in batch]

    # Handle retrieved texts (if present)
    if 'retrieved_texts' in batch[0]:
        retrieved_texts = [item['retrieved_texts'] for item in batch]
    else:
        retrieved_texts = None
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'text': texts,
        'retrieved_texts': retrieved_texts
    }


if __name__ == '__main__':
    # Test dataset
    print("Testing OPT Dataset...")
    
    dataset = MIMICCXRDataset(
        csv_file='./data/processed/mimic_cxr_train.csv',
        tokenizer_name='Salesforce/blip2-opt-2.7b'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test one sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['pixel_values'].shape}")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")
    print(f"  Text: {sample['text'][:100]}...")
    
    print("\n✓ Dataset test passed!")
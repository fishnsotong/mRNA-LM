#!/usr/bin/env python3

import os
import torch
import numpy as np
import pandas as pd
import argparse
from Bio import SeqIO
from scipy.special import softmax

from transformers import BertForSequenceClassification, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing

from peft import PeftModel, LoraConfig

def build_tokenizer(region="cds"):
    """Build the same tokenizer used during training"""
    lst_ele = list('AUGC')
    lst_voc = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    
    if region == "cds":
        # Generate all possible codons (4^3 = 64 combinations)
        for a1 in lst_ele:
            for a2 in lst_ele:
                for a3 in lst_ele:
                    lst_voc.extend([f'{a1}{a2}{a3}'])
    else:
        # For 5' UTR and 3' UTR, use single nucleotides
        for a1 in lst_ele:
            lst_voc.extend([f'{a1}'])
                    
    dic_voc = dict(zip(lst_voc, range(len(lst_voc))))
    tokenizer = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
    tokenizer.add_special_tokens(['[PAD]','[CLS]', '[UNK]', '[SEP]','[MASK]'])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = BertProcessing(
        ("[SEP]", dic_voc['[SEP]']),
        ("[CLS]", dic_voc['[CLS]']),
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, 
        unk_token='[UNK]',
        sep_token='[SEP]',
        pad_token='[PAD]',
        cls_token='[CLS]',
        mask_token='[MASK]'
    )

def tokenize_sequence(seq, kmer_len=3, stride=3):
    """Convert DNA/RNA sequence to space-separated kmers (same as training)"""
    seq = seq.upper().replace("T", "U")  # Convert T to U for RNA
    kmer_list = []
    for j in range(0, (len(seq) - kmer_len) + 1, stride):
        kmer_list.append(seq[j:j + kmer_len])
    return " ".join(kmer_list)

def load_model_and_adapter(base_model_path, adapter_path, num_labels=5):
    """Load the base model and apply the trained LoRA adapter"""
    
    # Load base model
    model = BertForSequenceClassification.from_pretrained(
        base_model_path, 
        num_labels=num_labels
    )
    
    # Build tokenizer (must match training tokenizer)
    tokenizer = build_tokenizer("cds")
    
    # Resize embeddings if needed
    if model.config.vocab_size != len(tokenizer):
        print(f"Resizing embeddings: model vocab {model.config.vocab_size} -> tokenizer vocab {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Set to evaluation mode
    model.eval()
    
    return model, tokenizer

def predict_sequences(model, tokenizer, sequences, batch_size=32, max_length=1024):
    """Make predictions on a list of sequences"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    predictions = []
    probabilities = []
    
    # Process in batches
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        
        # Tokenize sequences
        tokenized_seqs = [tokenize_sequence(seq) for seq in batch_seqs]
        
        # Encode with tokenizer
        encoded = tokenizer(
            tokenized_seqs,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            
            # Get predicted classes
            batch_predictions = np.argmax(logits, axis=-1)
            predictions.extend(batch_predictions.tolist())
            
            # Get probabilities
            batch_probs = softmax(logits, axis=1)
            probabilities.extend(batch_probs.tolist())
    
    return predictions, probabilities

def read_fasta(fasta_path):
    """Read sequences from FASTA file"""
    sequences = []
    sequence_ids = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        sequence_ids.append(record.id)
    
    return sequences, sequence_ids

def main():
    parser = argparse.ArgumentParser(description='Inference script for CDS classification')
    parser.add_argument('--fasta', '-f', type=str, required=True, 
                       help='Path to FASTA file containing CDS sequences')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to checkpoint directory (e.g., cds_model_class_eval0_test1/checkpoint-1984)')
    parser.add_argument('--base_model', '-b', type=str, default='codonbert',
                       help='Path to base model (default: codonbert)')
    parser.add_argument('--output', '-o', type=str, default='predictions.csv',
                       help='Output CSV file path')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device number')
    
    args = parser.parse_args()
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    
    print("Loading model and adapter...")
    model, tokenizer = load_model_and_adapter(
        base_model_path=args.base_model,
        adapter_path=args.checkpoint,
        num_labels=5  # Based on protein expression class task
    )
    
    print("Reading FASTA file...")
    sequences, sequence_ids = read_fasta(args.fasta)
    print(f"Loaded {len(sequences)} sequences")
    
    print("Making predictions...")
    predictions, probabilities = predict_sequences(
        model, tokenizer, sequences, batch_size=args.batch_size
    )
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'sequence_id': sequence_ids,
        'predicted_class': predictions,
        'class_0_prob': [p[0] for p in probabilities],
        'class_1_prob': [p[1] for p in probabilities],
        'class_2_prob': [p[2] for p in probabilities],
        'class_3_prob': [p[3] for p in probabilities],
        'class_4_prob': [p[4] for p in probabilities],
        'max_probability': [max(p) for p in probabilities]
    })
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    
    # Print summary
    print("\nPrediction Summary:")
    print(results_df['predicted_class'].value_counts().sort_index())
    print(f"\nMean confidence: {results_df['max_probability'].mean():.3f}")

if __name__ == "__main__":
    main()
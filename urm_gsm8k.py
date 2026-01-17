"""
Universal Reasoning Model (URM) - GSM-8K Implementation
Based on paper: https://arxiv.org/abs/2512.14693

Key innovations from paper:
1. Recurrent transformer layers (Universal Transformer)
2. ConvSwiGLU for local token interactions
3. Truncated Backpropagation Through Loops (TBPTL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
from tqdm import tqdm
import json

# ==================== Model Components ====================

class ConvSwiGLU(nn.Module):
    """ConvSwiGLU: SwiGLU with depthwise convolution for local interactions"""
    def __init__(self, d_model, d_ff, kernel_size=2):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        
        # Depthwise 1D convolution
        self.conv = nn.Conv1d(
            d_ff, d_ff, 
            kernel_size=kernel_size,
            padding=kernel_size-1,
            groups=d_ff,  # depthwise
            bias=False
        )
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        gate = self.w_gate(x)
        up = self.w_up(x)
        
        # Apply SiLU gating
        gated = F.silu(gate) * up
        
        # Apply depthwise convolution
        # [batch, seq_len, d_ff] -> [batch, d_ff, seq_len]
        gated_t = gated.transpose(1, 2)
        conv_out = self.conv(gated_t)
        # Trim padding and transpose back
        conv_out = conv_out[:, :, :-1].transpose(1, 2)
        
        # Project back down
        return self.w_down(conv_out)


class URMLayer(nn.Module):
    """Single URM transformer layer with ConvSwiGLU"""
    def __init__(self, d_model=256, n_heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # ConvSwiGLU feedforward
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = ConvSwiGLU(d_model, d_ff)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)
        
        # Feedforward with residual
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + self.dropout(ff_out)
        
        return x


class UniversalReasoningModel(nn.Module):
    """
    URM: Universal Transformer with recurrent loops
    
    Key features:
    - Shared transformer layers applied multiple times (loops)
    - Truncated backpropagation through loops (TBPTL)
    - ConvSwiGLU for enhanced nonlinearity
    """
    def __init__(
        self, 
        vocab_size=50257,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=2,
        n_loops=6,
        max_seq_len=512,
        dropout=0.1,
        tbptt_window=3  # Truncated BPTT window
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_loops = n_loops
        self.tbptt_window = tbptt_window
        
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.register_buffer(
            'pos_encoding',
            self._create_positional_encoding(max_seq_len, d_model)
        )
        
        # Shared transformer layers (Universal Transformer concept)
        self.layers = nn.ModuleList([
            URMLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (common practice)
        self.output.weight = self.token_emb.weight
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _create_positional_encoding(self, max_len, d_model):
        """Sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with recurrent loops
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.token_emb(input_ids)
        x = x + self.pos_encoding[:seq_len]
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )
        
        # Recurrent loops (Universal Transformer)
        for loop in range(self.n_loops):
            # Truncated BPTT: detach gradients for earlier loops
            if loop < (self.n_loops - self.tbptt_window):
                x = x.detach()
            
            # Apply all layers in sequence
            for layer in self.layers:
                x = layer(x, mask=causal_mask)
        
        # Output projection
        x = self.norm(x)
        logits = self.output(x)
        
        return logits


# ==================== Dataset ====================

class GSM8KDataset(Dataset):
    """GSM-8K dataset with simple formatting"""
    def __init__(self, split='train', tokenizer=None, max_length=512, num_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load from HuggingFace
        dataset = load_dataset('openai/gsm8k', 'main', split=split)
        
        # Subsample if requested
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        self.data = []
        for item in dataset:
            # Format: "Question: {q}\nAnswer: {a}"
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            self.data.append(text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Labels are input_ids shifted (for next token prediction)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore last position
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# ==================== Training ====================

def train_urm(
    model,
    train_loader,
    val_loader=None,
    epochs=3,
    lr=1e-4,
    device='cuda',
    save_path='urm_gsm8k.pt'
):
    """Train URM on GSM-8K"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # Cosine learning rate schedule
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            logits = model(input_ids)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} - Train Loss: {avg_loss:.4f}')
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    
                    logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}')
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model to {save_path}')
    
    return model


def generate_answer(model, tokenizer, question, device='cuda', max_new_tokens=256):
    """Generate answer for a question"""
    model.eval()
    
    # Format input
    text = f"Question: {question}\nAnswer:"
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated.split("Answer:")[-1].strip()


# ==================== Main ====================

def main():
    """Main training script"""
    print("=== URM for GSM-8K ===")
    print("Loading tokenizer...")
    
    # Use GPT-2 tokenizer (simple and effective)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = GSM8KDataset('train', tokenizer, num_samples=100)
    test_dataset = GSM8KDataset('test', tokenizer, num_samples=20)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("Creating URM model...")
    model = UniversalReasoningModel(
        vocab_size=len(tokenizer),
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=2,
        n_loops=6,  # Key URM parameter
        max_seq_len=512,
        dropout=0.1,
        tbptt_window=3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = train_urm(
        model,
        train_loader,
        test_loader,
        epochs=5,
        lr=1e-4,
        device=device,
        save_path='urm_gsm8k_best.pt'
    )
    
    # Test generation
    print("\n=== Testing Generation ===")
    test_questions = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. How much more money does Betty need?",
    ]
    
    for q in test_questions:
        print(f"\nQ: {q}")
        answer = generate_answer(model, tokenizer, q, device)
        print(f"A: {answer}")
    
    print("\n=== Training Complete ===")


if __name__ == '__main__':
    main()

"""
Train a 124M parameter decoder-only transformer model on Shakespeare text.
Target: Loss < 0.099999
"""
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class DataLoaderLite:
    def __init__(self, B, T, split='train'):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


def get_lr(it, warmup_iters, max_iters, max_lr, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def main():
    # Device setup
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Hyperparameters - OPTIMIZED FOR GPU
    batch_size = 32  # Larger batch for GPU
    sequence_length = 256  # Longer context for better learning  
    max_iters = 30000  # Need more steps to reach target
    warmup_iters = 200
    max_lr = 6e-4  # Standard GPT-2 learning rate
    min_lr = max_lr * 0.1
    grad_clip = 1.0  # Gradient clipping
    eval_interval = 50  # Print every 50 steps
    save_interval = 1000
    target_loss = 0.099999

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Tokens per batch: {batch_size * sequence_length}")
    print(f"  Max iterations: {max_iters}")
    print(f"  Target loss: {target_loss}\n")

    # Model setup
    print("Initializing model...")
    model = GPT(GPTConfig())
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Data loader
    train_loader = DataLoaderLite(B=batch_size, T=sequence_length)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), eps=1e-8)

    # Training loop
    print("\nStarting training...")
    print("-" * 80)

    for step in range(max_iters):
        t0 = time.time()

        # Get learning rate for this iteration
        lr = get_lr(step, warmup_iters, max_iters, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits, loss = model(x, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update
        optimizer.step()

        # Timing
        t1 = time.time()
        dt = (t1 - t0) * 1000  # time in milliseconds

        # Logging
        if step % eval_interval == 0 or step == max_iters - 1:
            print(f"step {step:4d} | loss: {loss.item():.6f} | lr: {lr:.2e} | time: {dt:.2f}ms")

        # Check if we've reached target loss
        if loss.item() < target_loss:
            print("\n" + "=" * 80)
            print(f"🎉 TARGET LOSS REACHED! 🎉")
            print(f"Step: {step}")
            print(f"Loss: {loss.item():.6f}")
            print(f"Target: {target_loss}")
            print("=" * 80)
            
            # Save the model
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'loss': loss.item(),
                'config': model.config,
            }
            torch.save(checkpoint, 'model_checkpoint_final.pt')
            print("Model saved to 'model_checkpoint_final.pt'")
            break

        # Periodic checkpoint saving
        if step > 0 and step % save_interval == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'loss': loss.item(),
                'config': model.config,
            }
            torch.save(checkpoint, f'checkpoint_step_{step}.pt')
            print(f"Checkpoint saved at step {step}")

    print("\nTraining complete!")
    print(f"Final loss: {loss.item():.6f}")

    # Generate some sample text
    print("\n" + "=" * 80)
    print("Generating sample text...")
    print("=" * 80)
    
    model.eval()
    enc = tiktoken.get_encoding('gpt2')
    
    # Generate from a few different prompts
    prompts = ["First Citizen:", "ROMEO:", "To be or not"]
    
    with torch.no_grad():
        for prompt in prompts:
            tokens = enc.encode(prompt)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            # Generate
            for _ in range(50):
                logits, _ = model(tokens)
                logits = logits[:, -1, :]  # Get last token logits
                probs = F.softmax(logits, dim=-1)
                
                # Sample from top-k
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                next_token = torch.gather(topk_indices, -1, ix)
                
                tokens = torch.cat([tokens, next_token], dim=1)
            
            # Decode and print
            generated_text = enc.decode(tokens[0].tolist())
            print(f"\nPrompt: '{prompt}'")
            print(f"Generated: {generated_text}")
            print("-" * 80)


if __name__ == "__main__":
    main()


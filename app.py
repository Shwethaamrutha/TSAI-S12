"""
HuggingFace Spaces App for GPT-2 124M Shakespeare Model
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import gradio as gr
import math
from dataclasses import dataclass


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
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
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


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
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# Load model
print("Loading model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig()
model = GPT(config)

model_loaded = False

# Try to load model from HuggingFace Model Hub first, then local file
try:
    from huggingface_hub import hf_hub_download
    import os
    
    # Try to get model path from environment variable or use default
    repo_id = os.getenv('HF_MODEL_REPO', 'shwethd/gpt2-shakespeare-124m')
    
    try:
        print(f"Attempting to load from HuggingFace Hub: {repo_id}")
        
        # Try SafeTensors first (more secure, no pickle issues)
        try:
            from safetensors.torch import load_file
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="model.safetensors",
                    cache_dir=None
                )
                state_dict = load_file(model_path, device=device)
                model.load_state_dict(state_dict)
                # Restore weight sharing (broken during SafeTensors conversion)
                # lm_head.weight and transformer.wte.weight should share memory
                model.transformer.wte.weight = model.lm_head.weight
                model_loaded = True
                print(f"✅ Model loaded successfully from SafeTensors: {repo_id}")
            except Exception as e:
                print(f"SafeTensors not found ({e}), trying .pt file...")
                # Fallback to .pt file
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="model_checkpoint_final.pt",
                    cache_dir=None
                )
                # PyTorch 2.6+ requires weights_only=False for custom classes
                # This is safe since we trust our own trained model
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # If checkpoint is the state dict itself
                    model.load_state_dict(checkpoint)
                
                model_loaded = True
                print(f"✅ Model loaded successfully from HuggingFace Hub: {repo_id}")
        except ImportError:
            # safetensors not installed, use .pt file
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename="model_checkpoint_final.pt",
                cache_dir=None
            )
            # PyTorch 2.6+ requires weights_only=False for custom classes
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # If checkpoint is the state dict itself
                model.load_state_dict(checkpoint)
            
            model_loaded = True
            print(f"✅ Model loaded successfully from HuggingFace Hub: {repo_id}")
    except Exception as e:
        print(f"⚠️ Could not load from Hub ({e}), trying local file...")
        try:
            # Fallback to local file
            # PyTorch 2.6+ requires weights_only=False for custom classes
            checkpoint = torch.load('model_checkpoint_final.pt', map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model_loaded = True
            print("✅ Model loaded from local checkpoint")
        except Exception as e2:
            print(f"❌ Could not load from local file either: {e2}")
except FileNotFoundError:
    print("❌ Warning: Model checkpoint not found. Using untrained model.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️ Using untrained model as fallback - output will be random!")

if not model_loaded:
    print("⚠️ WARNING: Model is using random weights! Generation will be nonsensical.")
    print("Please ensure model_checkpoint_final.pt is uploaded to HuggingFace Model Hub.")

model.to(device)
model.eval()
print(f"Model ready on {device}")

enc = tiktoken.get_encoding('gpt2')


def generate_text(prompt, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1):
    """Generate text from prompt with improved sampling"""
    try:
        if not model_loaded:
            return "❌ Error: Model not loaded correctly. Please check that model_checkpoint_final.pt is uploaded to HuggingFace Model Hub (shwethd/gpt2-shakespeare-124m)."
        
        # Validate inputs
        if not prompt or len(prompt.strip()) == 0:
            return "Please enter a prompt."
        
        temperature = max(0.1, min(2.0, temperature))  # Clamp temperature
        top_k = max(1, min(100, int(top_k)))  # Clamp top_k
        top_p = max(0.1, min(1.0, float(top_p)))  # Clamp top_p (nucleus sampling)
        repetition_penalty = max(1.0, min(1.5, float(repetition_penalty)))  # Clamp repetition penalty
        max_new_tokens = max(1, min(200, int(max_new_tokens)))  # Clamp max tokens
        
        # Encode prompt
        tokens = enc.encode(prompt)
        if len(tokens) == 0:
            return "Error: Could not encode prompt."
        
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate with improved sampling strategy
        with torch.no_grad():
            # Track recent tokens for repetition penalty
            recent_tokens = set()
            
            for i in range(max_new_tokens):
                # Forward pass
                logits, _ = model(tokens)
                logits = logits[:, -1, :] / max(temperature, 0.1)  # Apply temperature
                
                # Apply repetition penalty to reduce loops
                if repetition_penalty > 1.0 and len(recent_tokens) > 0:
                    for token_id in recent_tokens:
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Apply top-p (nucleus) sampling first - often better than just top-k
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[..., 0] = False
                    
                    # Create mask
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0
                    
                    # Renormalize
                    probs = probs / probs.sum()
                
                # Apply top-k filtering (after top-p for better quality)
                if top_k < logits.size(-1):
                    topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                    # Create filtered probabilities
                    filtered_probs = torch.zeros_like(probs)
                    filtered_probs.scatter_(-1, topk_indices, topk_probs)
                    # Renormalize
                    filtered_probs = filtered_probs / filtered_probs.sum()
                    probs = filtered_probs
                
                # Avoid NaN or zero probabilities
                if torch.isnan(probs).any() or (probs.sum() == 0):
                    probs = torch.ones_like(probs) / probs.size(-1)
                
                # Sample from distribution
                next_token = torch.multinomial(probs, 1)
                
                # Update recent tokens for repetition penalty (keep last 20 tokens)
                token_id = next_token.item()
                recent_tokens.add(token_id)
                if len(recent_tokens) > 20:
                    # Remove oldest tokens (simple approach: keep last 20)
                    recent_tokens = set(list(recent_tokens)[-20:])
                
                # Append to sequence
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # Early stopping: stop if we generate end-of-text token (if present)
                # For GPT-2 tokenizer, we can check for certain patterns
                if tokens.size(1) >= config.block_size:
                    break
        
        # Decode
        generated_text = enc.decode(tokens[0].tolist())
        
        # Post-process to fix spacing issues (common with BPE tokenizers)
        import re
        
        # Fix 0: Remove the prompt from the beginning if it appears as a speaker name
        # This handles cases where user enters "First Citizen:" and model repeats it
        # Normalize prompt: remove colon, strip, convert to uppercase for comparison
        prompt_normalized = prompt.strip().replace(':', '').strip().upper()
        
        # Process all lines to find and remove prompt matches
        lines = generated_text.split('\n')
        cleaned_lines = []
        prompt_removed = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines at the start (but only if we haven't added any content yet)
            if not line_stripped:
                if not cleaned_lines:
                    continue  # Skip leading empty lines
                else:
                    cleaned_lines.append(line)  # Keep empty lines after content starts
                    continue
            
            # Normalize line for comparison (remove colon, case-insensitive)
            line_normalized = line_stripped.replace(':', '').strip().upper()
            
            # Check if this line matches the prompt (case-insensitive, allowing for colon)
            # Check if it's a speaker name format (all caps OR title case OR mixed case)
            is_speaker_line = (re.match(r'^([A-Z][A-Z\s]+?):\s*$', line_stripped) or  # All caps: "FIRST CITIZEN:"
                              re.match(r'^([A-Z][a-z]+(?:\s+[a-zA-Z]+)+):\s*$', line_stripped) or  # Title case: "First Citizen:"
                              re.match(r'^([A-Z][A-Za-z\s]+?):\s*$', line_stripped))  # Mixed case: "First Citizen:" or "FIRST Citizen:"
            
            # If this line matches the prompt (case-insensitive), remove it
            # Be more aggressive: if it matches the prompt, remove it even if pattern doesn't match exactly
            if line_normalized == prompt_normalized and not prompt_removed:
                # Additional check: if it ends with colon, it's likely a speaker name
                if line_stripped.endswith(':'):
                    # This is the prompt appearing as a speaker - skip it
                    prompt_removed = True
                    continue
                # Also remove if it's a speaker line pattern
                elif is_speaker_line:
                    prompt_removed = True
                    continue
            
            # If we've already removed the prompt, add the line
            cleaned_lines.append(line)
        
        generated_text = '\n'.join(cleaned_lines)
        
        # If after removing prompt, first line is orphaned dialogue (no speaker), handle it
        # Keep removing orphaned dialogue at the start until we find a speaker or valid content
        # Limit to max 10 iterations to avoid infinite loops
        lines = generated_text.split('\n')
        start_idx = 0
        max_iterations = 10
        iteration = 0
        
        while start_idx < len(lines) and iteration < max_iterations:
            iteration += 1
            first_line = lines[start_idx].strip() if start_idx < len(lines) else ''
            
            if not first_line:
                # Remove empty first line
                start_idx += 1
                continue
            
            # Check if first line is a speaker name
            is_speaker = re.match(r'^([A-Z][A-Z\s]+?):\s*$', first_line) or \
                        re.match(r'^([A-Z][a-z]+(?:\s+[a-zA-Z]+)+):\s*$', first_line)
            
            if is_speaker:
                # Found a speaker, stop removing
                break
            
            # Check if it's orphaned dialogue (starts with capital, has punctuation, but no speaker)
            if re.match(r'^[A-Z]', first_line) and ('.' in first_line or ',' in first_line or '!' in first_line or '?' in first_line):
                # Remove the orphaned first line
                start_idx += 1
            else:
                # Not clearly orphaned dialogue, stop removing
                break
        
        generated_text = '\n'.join(lines[start_idx:])
        
        # Fix 1: lowercase followed by uppercase (e.g., "perpetualWith" -> "perpetual With", "AOr" -> "A Or")
        generated_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', generated_text)
        # Also fix single letter + capital word (e.g., "AOr" -> "A Or")
        generated_text = re.sub(r'\b([A-Z])([A-Z][a-z]+)', r'\1 \2', generated_text)
        
        # Fix 1b: Fix spacing issues like "furt her" -> "further", "T his" -> "This", "y our" -> "your", "th at" -> "that"
        # OPTIMIZED: Only process most common split words to reduce computation
        # Focus on words that are most likely to be split incorrectly
        common_words_fix = [
            'further', 'this', 'that', 'there', 'where', 'here', 'their', 'your', 'our', 
            'man', 'men', 'woman', 'women', 'content', 'gentle', 'gently',
            'house', 'made', 'lost', 'rough', 'see', 'might', 'any', 'one',
            'well', 'too', 'him', 'her', 'them', 'they', 'the', 'and', 'but',
            'for', 'not', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will',
            'shall', 'would', 'could', 'should', 'be', 'is', 'it', 'he', 'she', 'we',
            'you', 'me', 'my', 'his', 'into', 'onto', 'upon', 'within', 'without',
            'together', 'honour', 'honor', 'common', 'complain', 'again', 'apparent'
        ]
        # Pre-compile patterns for common splits (only most common 2-3 splits per word)
        for word in common_words_fix:
            word_lower = word.lower()
            # Only try 2-3 most common split positions (middle, quarter, three-quarter)
            split_positions = []
            if len(word_lower) > 2:
                split_positions = [len(word_lower) // 2]  # Most common: middle split
                if len(word_lower) > 4:
                    split_positions.append(len(word_lower) // 4)
                    split_positions.append(3 * len(word_lower) // 4)
            
            for i in split_positions:
                if i < 1 or i >= len(word_lower):
                    continue
                first_part = word_lower[:i]
                second_part = word_lower[i:]
                
                # Combined pattern with case-insensitive flag (more efficient)
                pattern = r'\b' + re.escape(first_part) + r'\s+' + re.escape(second_part) + r'\b'
                generated_text = re.sub(pattern, word, generated_text, flags=re.IGNORECASE)
        
        # Fix 2: Common word boundaries that got merged (e.g., "perpetualwith" -> "perpetual with")
        # Add space before common words that might have been merged
        common_words = ['with', 'the', 'and', 'that', 'this', 'have', 'from', 'not', 'but', 'for', 'are', 'was', 'were', 'been', 'will', 'shall', 'would', 'could', 'should', 'be', 'your', 'you', 'our', 'my', 'his', 'her', 'their', 'him', 'them', 'to', 'of', 'in', 'on', 'at', 'as', 'is', 'it', 'he', 'she', 'we', 'they', 'an', 'a']
        for word in common_words:
            # Only add space if it's not already separated and follows a lowercase letter
            pattern = r'([a-z])(' + word + r'\b)'
            generated_text = re.sub(pattern, r'\1 \2', generated_text, flags=re.IGNORECASE)
        
        # Fix 2c: Fix double words (e.g., "but but" -> "but")
        generated_text = re.sub(r'\b(\w+)\s+\1\b', r'\1', generated_text, flags=re.IGNORECASE)
        
        # Fix 2d: Fix spacing after commas (e.g., "What,bear" -> "What, bear")
        generated_text = re.sub(r',([a-zA-Z])', r', \1', generated_text)
        
        # Fix 1c: Fix multiple splits in one word - OPTIMIZED: Only handle most common cases
        # Focus on very common words that are most likely to be split
        multi_split_words = ['count', 'your', 'our', 'the', 'and', 'but', 'for', 'not', 'are', 'was', 'were', 
                            'been', 'have', 'has', 'had', 'will', 'this', 'that', 'there', 'where', 'here', 
                            'their', 'what', 'common', 'complain', 'honour', 'honor', 'again', 'apparent']
        for word in multi_split_words:
            word_lower = word.lower()
            if len(word_lower) > 2:
                # Pattern 1: letter space letter space ... (all letters split individually) - only for short words
                if len(word_lower) <= 5:
                    letters = list(word_lower)
                    pattern_parts = [re.escape(letter) + r'\s+' for letter in letters[:-1]]
                    pattern_parts.append(re.escape(letters[-1]))
                    pattern = r'\b' + ''.join(pattern_parts) + r'\b'
                    generated_text = re.sub(pattern, word, generated_text, flags=re.IGNORECASE)
                
                # Pattern 2: Handle two-part splits - only try most common split (middle)
                split_pos = len(word_lower) // 2
                if split_pos > 0 and split_pos < len(word_lower):
                    first_part = word_lower[:split_pos]
                    second_part = word_lower[split_pos:]
                    pattern_2part = r'\b' + re.escape(first_part) + r'\s+' + re.escape(second_part) + r'\b'
                    generated_text = re.sub(pattern_2part, word, generated_text, flags=re.IGNORECASE)
        
        # Fix 2e: Fix merged words that should be separate (e.g., "himt" -> "him to", "incwold" -> "in cold")
        # Common patterns where words got merged incorrectly
        merged_fixes = [
            # Pronoun + "t" (likely "to" got merged)
            (r'\bhimt\s+', 'him to '),  # "himt me" -> "him to me"
            (r'\bhert\s+', 'her to '),  # "hert him" -> "her to him"
            (r'\bthemt\s+', 'them to '),  # "themt us" -> "them to us"
            (r'\byout\s+', 'you to '),  # "yout me" -> "you to me"
            (r'\bhimt([,.;:!?])', r'him to\1'),  # "himt," -> "him to,"
            (r'\bhert([,.;:!?])', r'her to\1'),
            (r'\bthemt([,.;:!?])', r'them to\1'),
            (r'\byout([,.;:!?])', r'you to\1'),
            # Other merged patterns
            (r'\bincwold\b', 'in cold'),  # "incwold" -> "in cold"
            (r'\bincold\b', 'in cold'),  # "incold" -> "in cold"
            (r'\blikeled\b', 'liked'),  # "likeled" -> "liked"
            (r'\bh\s+on\s+our\b', 'honour'),  # "h on our" -> "honour"
            (r'\bh\s+on\s+or\b', 'honor'),  # "h on or" -> "honor"
            (r'\bHapp\s+up\s+on\'t\b', "Happen upon't"),  # "Happ up on't" -> "Happen upon't"
            (r'\bhapp\s+up\s+on\'t\b', "happen upon't"),
            # Fix "comm on" -> "common" (if not already fixed)
            (r'\bcomm\s+on\b', 'common'),
            (r'\bComm\s+on\b', 'Common'),
            # Fix "compl a in" -> "complain" (multiple splits)
            (r'\bcompl\s+a\s+in\b', 'complain'),
            (r'\bCompl\s+a\s+in\b', 'Complain'),
            # Fix "As s he" -> "As she"
            (r'\bAs\s+s\s+he\b', 'As she'),
            (r'\bas\s+s\s+he\b', 'as she'),
            # Fix "ag a in" -> "again" (multiple splits)
            (r'\bag\s+a\s+in\b', 'again'),
            (r'\bAg\s+a\s+in\b', 'Again'),
            # Fix "UN TO" -> "UNTO" (before Fix 3c processes it)
            (r'\bUN\s+TO\b', 'UNTO'),
            (r'\bun\s+to\b', 'unto'),
            # Fix potential word issues
            (r'\bcoronured\b', 'crowned'),  # "coronured" -> "crowned"
            (r'\beyuls\b', 'evils'),  # "eyuls" -> "evils"
            # Fix "AOr" -> "A Or" or "Or" (if it's at start of sentence)
            (r'\bAOr\b', 'A Or'),
            (r'^A Or\s+', 'Or '),  # If "A Or" is at start, might just be "Or"
            # Fix "fe at" -> "feat"
            (r'\bfe\s+at\b', 'feat'),
            (r'\bFe\s+at\b', 'Feat'),
            # Fix "MORE TH AN HALF" -> "MORE THAN HALF" (but this might be dialogue, not speaker)
            (r'\bTH\s+AN\b', 'THAN'),
            (r'\bth\s+an\b', 'than'),
            # Fix "F IT" -> "FIT" (in all caps dialogue)
            (r'\bF\s+IT\b', 'FIT'),
            (r'\bf\s+it\b', 'fit'),
            (r'\bF\s+it\b', 'Fit'),
            # Fix "C A" -> "CA" (but be careful - might be part of "C A:" speaker name)
            # Actually, "C A:" should be merged to "CA:" or might be "CLARENCE:" - handle in speaker fix
            # Fix "OUCESTER" -> "GLOUCESTER" (missing "GL" prefix)
            (r'\bOUCESTER\b', 'GLOUCESTER'),
            (r'\bOucester\b', 'Gloucester'),
            # Fix "stuff'd" -> "stuffed" (if needed, but "stuff'd" is valid Shakespeare)
            # Actually, "stuff'd" is correct Shakespeare spelling, so we'll leave it
            # Fix duplicate words: "if it be it possible" -> "if it be possible"
            (r'\bif it be it\b', 'if it be'),
            (r'\bIf it be it\b', 'If it be'),
            # Fix duplicate "belike" -> remove one
            (r'\bbelike\s+that\s+you\s+were\s+right\s+gentle\s+exercise,\s+belike\b', 'belike that you were right gentle exercise'),
            (r'\bBelike\s+that\s+you\s+were\s+right\s+gentle\s+exercise,\s+belike\b', 'Belike that you were right gentle exercise'),
        ]
        for pattern, replacement in merged_fixes:
            generated_text = re.sub(pattern, replacement, generated_text, flags=re.IGNORECASE)
        
        # Fix 2f: Fix "content on" - this is likely two separate words, but ensure proper spacing
        generated_text = re.sub(r'\bcontenton\b', 'content on', generated_text, flags=re.IGNORECASE)
        
        # Fix 2g: Fix "toget her" -> "together"
        generated_text = re.sub(r'\btoget\s+her\b', 'together', generated_text, flags=re.IGNORECASE)
        
        # Fix 2b: Fix contractions that got merged (e.g., "You'llbe" -> "You'll be")
        # Add space after contractions before lowercase words
        contractions = ["'ll", "'ve", "'re", "'d", "'t", "'s", "'m"]
        for contraction in contractions:
            # Pattern: contraction followed by lowercase letter (e.g., "You'llbe" -> "You'll be")
            pattern = r"(" + re.escape(contraction) + r")([a-z])"
            generated_text = re.sub(pattern, r'\1 \2', generated_text, flags=re.IGNORECASE)
        
        # Fix 3: Fix split speaker names (e.g., "ALL ANC A:" -> "ALLANCA:", "GENTLEM AN:" -> "GENTLEMAN:")
        # Pattern: All caps words separated by spaces ending with colon (likely split speaker name)
        # First, try to merge split speaker names: "ALL ANC A:" -> "ALLANCA:", "GENTLEM AN:" -> "GENTLEMAN:"
        # But be careful - some speaker names might legitimately have spaces (e.g., "FIRST CITIZEN:")
        lines = generated_text.split('\n')
        fixed_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Check if line looks like a split speaker name (all caps, has spaces, ends with colon)
            # Pattern 1: Multiple all-caps words with spaces: "ALL ANC A:" or "GENTLEM AN:"
            if re.match(r'^([A-Z]+\s+[A-Z]+\s*[A-Z]*):\s*$', line_stripped):
                # Check if it's a known multi-word speaker name (keep those)
                known_multi_word_speakers = ['FIRST CITIZEN', 'SECOND CITIZEN', 'THIRD CITIZEN', 
                                            'FIRST GENTLEMAN', 'SECOND GENTLEMAN', 'THIRD GENTLEMAN',
                                            'FIRST SERVANT', 'SECOND SERVANT', 'LADY MACBETH',
                                            'KING HENRY', 'PRINCE HAMLET', 'DUKE VINCENTIO']
                is_known = False
                for known in known_multi_word_speakers:
                    if known in line_stripped.upper():
                        is_known = True
                        break
                
                if not is_known:
                    # Try to merge: "ALL ANC A:" -> "ALLANCA:", "GENTLEM AN:" -> "GENTLEMAN:", "C A:" -> "CA:" or "CLARENCE:"
                    # Remove spaces between all-caps words before colon
                    merged = re.sub(r'([A-Z]+)\s+([A-Z]+)\s*([A-Z]*):', r'\1\2\3:', line_stripped)
                    
                    # Special case: "C A:" might be "CLARENCE:" - check if it's a known pattern
                    if re.match(r'^C\s+A:\s*$', line_stripped):
                        # Check context - if it's near "Clarence" or "Sir Clarence", it's likely "CLARENCE:"
                        # For now, merge to "CA:" and let it be handled as a potential speaker
                        merged = 'CLARENCE:'
                    
                    # Only use merged if it makes sense (not too long, looks like a word)
                    if len(merged) < 30:  # Reasonable speaker name length
                        fixed_lines.append(merged)
                    else:
                        fixed_lines.append(line)
                else:
                    # Keep known multi-word speaker names as is
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        generated_text = '\n'.join(fixed_lines)
        
        # Fix 3b: Add space before character names (all caps words) and fix missing punctuation
        # First, fix cases like "Barn MENENIUS:" -> "Barn. MENENIUS:" or "Barn, MENENIUS:"
        # Pattern: lowercase word followed immediately by all-caps speaker name
        generated_text = re.sub(r'([a-z]+)([A-Z]{2,}):', r'\1. \2:', generated_text)
        # Then add space before character names
        generated_text = re.sub(r'([a-z])([A-Z]{2,})', r'\1 \2', generated_text)
        
        # Fix 3b: Normalize speaker names (e.g., "Romeo and juliet" -> "ROMEO AND JULIET:")
        # Handle mixed case speaker names that should be all caps
        # Also handle "First Citizen:" -> "FIRST CITIZEN:"
        lines = generated_text.split('\n')
        normalized_lines = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if line is a potential speaker name (title case or mixed case, 2+ words)
            # Pattern: "Romeo and juliet", "Romeo And Juliet", "First Citizen", etc.
            speaker_pattern = r'^([A-Z][a-z]+(?:\s+[a-zA-Z]+)+)\s*:?\s*$'
            match = re.match(speaker_pattern, line_stripped)
            
            # Also check for all-caps speaker names (already normalized)
            all_caps_speaker = re.match(r'^([A-Z][A-Z\s]+?):\s*$', line_stripped)
            
            if match:
                # Check if next line is dialogue (not another speaker)
                is_speaker = False
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # If next line is not empty and not a speaker name, this is likely a speaker
                    if next_line and not re.match(r'^([A-Z][A-Z\s]+?):\s*$', next_line):
                        is_speaker = True
                elif i == 0:  # First line is likely a speaker if it matches pattern
                    is_speaker = True
                
                if is_speaker:
                    # Convert to all caps and ensure colon
                    speaker_name = match.group(1).upper()
                    normalized_lines.append(speaker_name + ':')
                    continue
            elif all_caps_speaker:
                # Already all caps, just ensure it has colon
                speaker_name = all_caps_speaker.group(1).strip()
                if not line_stripped.endswith(':'):
                    normalized_lines.append(speaker_name + ':')
                else:
                    normalized_lines.append(line)
                continue
            
            normalized_lines.append(line)
        
        generated_text = '\n'.join(normalized_lines)
        
        # Fix 0b: Remove prompt again after normalization (in case it was normalized to all caps)
        # This handles cases where "First Citizen:" was normalized to "FIRST CITIZEN:"
        prompt_normalized = prompt.strip().replace(':', '').strip().upper()
        lines = generated_text.split('\n')
        cleaned_lines_after_norm = []
        prompt_removed_after_norm = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines at the start
            if not line_stripped and not cleaned_lines_after_norm:
                continue
            
            # Normalize line for comparison (remove colon, case-insensitive)
            line_normalized = line_stripped.replace(':', '').strip().upper()
            
            # Check if this line matches the prompt (case-insensitive, allowing for colon)
            # Also check if it's a speaker name format (all caps after normalization)
            is_speaker_line = re.match(r'^([A-Z][A-Z\s]+?):\s*$', line_stripped)
            
            if is_speaker_line and line_normalized == prompt_normalized and not prompt_removed_after_norm:
                # This is the prompt appearing as a speaker - skip it
                prompt_removed_after_norm = True
                continue
            
            # If we've already removed the prompt, add the line
            cleaned_lines_after_norm.append(line)
        
        generated_text = '\n'.join(cleaned_lines_after_norm)
        
        # Fix 3c: Fix dialogue that was incorrectly formatted as speaker names
        # Pattern: All caps lines ending with colon that are actually dialogue (not speakers)
        # Examples: "HENCE ARE YOUR HONOUR TO ENTER:" -> "HENCE ARE YOUR HONOUR TO ENTER."
        #           "THERE SHOULD RUE:" -> "THERE SHOULD RUE."
        #           "UN TO THE LADY GREY:" -> "UNTO THE LADY GREY."
        # These are usually long phrases (3+ words) that don't look like character names
        lines = generated_text.split('\n')
        fixed_dialogue_lines = []
        # Known speaker names (keep these as speakers)
        known_speakers = ['BAPTISTA', 'GLOUCESTER', 'CLARENCE', 'ROMEO', 'JULIET', 'HAMLET', 'MACBETH', 
                         'KING', 'QUEEN', 'DUKE', 'PRINCE', 'LADY', 'FIRST', 'SECOND', 'THIRD',
                         'CITIZEN', 'GENTLEMAN', 'SERVANT', 'MENENIUS', 'COMINIUS', 'CORIOLANUS',
                         'VINCENTIO', 'ANGELO', 'ISABELLA', 'OTHELLO', 'DESDEMONA', 'IAGO']
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            # Check if line looks like all-caps speaker but is actually dialogue
            # Pattern: All caps, ends with colon
            if re.match(r'^([A-Z][A-Z\s]+?):\s*$', line_stripped):
                words = line_stripped.split()
                speaker_name = words[0] if words else ''
                
                # Check if it's a known speaker name (1-2 words, known name)
                is_known_speaker = (len(words) <= 2 and speaker_name in known_speakers) or \
                                  (len(words) == 2 and words[0] in ['FIRST', 'SECOND', 'THIRD'] and words[1] in ['CITIZEN', 'GENTLEMAN', 'SERVANT'])
                
                if is_known_speaker:
                    # Keep as speaker name
                    fixed_dialogue_lines.append(line)
                # If it has 3+ words, it's likely dialogue, not a speaker name
                elif len(words) >= 3:
                    # Convert colon to period (dialogue ending)
                    dialogue = line_stripped[:-1] + '.'  # Remove colon, add period
                    fixed_dialogue_lines.append(dialogue)
                # Also check if it contains common dialogue words (not speaker names)
                elif any(word in ['ARE', 'YOUR', 'HONOUR', 'TO', 'ENTER', 'SHOULD', 'RUE', 'THE', 'GREY', 'HENCE', 'THERE', 'UN', 'UNTIL', 'UNTO', 'MORE', 'THAN', 'HALF', 'TH', 'AN'] for word in words):
                    # Likely dialogue, not speaker
                    dialogue = line_stripped[:-1] + '.'  # Remove colon, add period
                    fixed_dialogue_lines.append(dialogue)
                # Special case: Single letter "A:" is likely dialogue or incomplete, not a speaker
                elif len(words) == 1 and words[0] == 'A':
                    # Convert to dialogue
                    fixed_dialogue_lines.append('A.')
                # Special case: "MORE THAN HALF:" is dialogue, not speaker
                elif 'MORE' in words and 'THAN' in words:
                    dialogue = line_stripped[:-1] + '.'  # Remove colon, add period
                    fixed_dialogue_lines.append(dialogue)
                else:
                    # Keep as speaker name (might be a short unknown character name)
                    fixed_dialogue_lines.append(line)
            else:
                fixed_dialogue_lines.append(line)
        
        generated_text = '\n'.join(fixed_dialogue_lines)
        
        # Fix 4: Remove duplicate speaker names (e.g., "EDWARD IV:\n...\nEDWARD IV:" -> keep only first)
        # More aggressive: remove same speaker if it appears within 5 lines (expanded window for empty lines)
        # Also handle case-insensitive duplicates (e.g., "First Citizen:" and "FIRST CITIZEN:")
        lines = generated_text.split('\n')
        cleaned_lines = []
        speaker_history = []  # Track recent speakers with their line numbers (case-insensitive)
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            # Check if this line is a speaker name (all caps after normalization)
            speaker_match = re.match(r'^([A-Z][A-Z\s]+?):\s*$', line_stripped)
            
            if speaker_match:
                speaker = speaker_match.group(1).strip()
                speaker_upper = speaker.upper()  # For case-insensitive comparison
                
                # Check if this speaker appeared recently (within last 5 lines - expanded for empty lines)
                # Check both exact match and case-insensitive match
                recent_speaker = False
                for hist_speaker, hist_line_num in speaker_history[-5:]:  # Check last 5 speakers
                    hist_speaker_upper = hist_speaker.upper()
                    if speaker == hist_speaker or speaker_upper == hist_speaker_upper:
                        recent_speaker = True
                        break
                
                if recent_speaker:
                    # Skip this duplicate speaker
                    continue
                
                # Add to history (store uppercase version for consistent comparison)
                speaker_history.append((speaker_upper, i))
                # Keep only last 15 speakers in history (expanded)
                if len(speaker_history) > 15:
                    speaker_history.pop(0)
                
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        generated_text = '\n'.join(cleaned_lines)
        
        # Fix 5: Remove speaker names with no dialogue (e.g., "KING:\nEDWARD IV:" -> "EDWARD IV:", "First Citizen:\n\nCLARENCE:" -> "CLARENCE:")
        # A speaker name should be followed by actual dialogue, not immediately by another speaker or empty lines
        lines = generated_text.split('\n')
        final_lines = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            speaker_match = re.match(r'^([A-Z][A-Z\s]+?):\s*$', line_stripped)
            
            if speaker_match:
                # Check if next non-empty line is another speaker or if there's no dialogue at all
                has_dialogue = False
                # Check up to 5 lines ahead (more generous to catch dialogue)
                for j in range(i + 1, min(i + 6, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:  # Skip empty lines
                        continue
                    # If next non-empty line is NOT a speaker, we have dialogue
                    if not re.match(r'^([A-Z][A-Z\s]+?):\s*$', next_line):
                        has_dialogue = True
                        break
                    # If next non-empty line IS a speaker, this speaker has no dialogue
                    elif re.match(r'^([A-Z][A-Z\s]+?):\s*$', next_line):
                        # This speaker has no dialogue - skip it
                        has_dialogue = False
                        break
                
                if not has_dialogue:
                    # This speaker has no dialogue, skip it
                    continue
            
            final_lines.append(line)
        
        generated_text = '\n'.join(final_lines)
        
        # Fix 5b: Fix merged text issues (e.g., "You?A:" -> "You? A:")
        # Add space after question/exclamation marks before capital letters
        generated_text = re.sub(r'([?!])([A-Z])', r'\1 \2', generated_text)
        
        # Fix 6: Remove multiple empty lines between speaker and dialogue
        generated_text = re.sub(r'([A-Z][A-Z\s]+?):\s*\n\s*\n+', r'\1:\n', generated_text)
        
        # Fix 7: Remove any remaining consecutive duplicate speakers (final cleanup)
        # Handle both exact duplicates and case-insensitive duplicates
        # This handles cases like "FIRST CITIZEN:\n\nFIRST CITIZEN:" -> "FIRST CITIZEN:"
        lines = generated_text.split('\n')
        final_cleaned_lines = []
        last_speaker_upper = None
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            speaker_match = re.match(r'^([A-Z][A-Z\s]+?):\s*$', line_stripped)
            
            if speaker_match:
                speaker = speaker_match.group(1).strip()
                speaker_upper = speaker.upper()
                
                # If this is the same speaker as the last one (case-insensitive), skip it
                if speaker_upper == last_speaker_upper:
                    continue
                
                last_speaker_upper = speaker_upper
                final_cleaned_lines.append(line)
            else:
                # Reset speaker tracking on non-speaker lines (but keep last_speaker for nearby duplicates)
                # Only reset if we have substantial content (not just empty lines)
                if line_stripped:  # Non-empty line
                    # Keep last_speaker for a few lines in case of empty lines
                    pass
                final_cleaned_lines.append(line)
        
        generated_text = '\n'.join(final_cleaned_lines)
        
        # Fix 7b: Remove duplicate words in sentences (e.g., "if it be it possible" -> "if it be possible")
        # Pattern: word followed by same word (case-insensitive)
        # But be careful not to remove valid repetitions like "very very" or "more more"
        # Only remove common function words that shouldn't repeat
        duplicate_word_patterns = [
            (r'\b(it)\s+\1\b', r'\1'),  # "it it" -> "it"
            (r'\b(the)\s+\1\b', r'\1'),  # "the the" -> "the"
            (r'\b(a)\s+\1\b', r'\1'),  # "a a" -> "a"
            (r'\b(an)\s+\1\b', r'\1'),  # "an an" -> "an"
            (r'\b(is)\s+\1\b', r'\1'),  # "is is" -> "is"
            (r'\b(was)\s+\1\b', r'\1'),  # "was was" -> "was"
            (r'\b(are)\s+\1\b', r'\1'),  # "are are" -> "are"
            (r'\b(be)\s+\1\b', r'\1'),  # "be be" -> "be"
        ]
        for pattern, replacement in duplicate_word_patterns:
            generated_text = re.sub(pattern, replacement, generated_text, flags=re.IGNORECASE)
        
        # Fix 8: Handle incomplete termination - remove incomplete words/sentences at the end
        # This happens when the model hits the token limit mid-generation
        if generated_text.strip():
            # Remove incomplete word at the end (word that doesn't end with punctuation or space)
            # Pattern: ends with a word that has no trailing punctuation/space
            # But keep if it ends with proper punctuation (. ! ? , ; :)
            lines = generated_text.split('\n')
            if lines:
                last_line = lines[-1].strip()
                
                # If last line doesn't end with punctuation and is not a speaker name
                if last_line and not re.match(r'^([A-Z][A-Z\s]+?):\s*$', last_line):
                    # Check if it ends with incomplete word (no punctuation, not a complete sentence)
                    # Remove if it ends with a word that looks incomplete
                    # Pattern: ends with word that has no punctuation
                    if not re.search(r'[.!?,;:]$', last_line):
                        # Check if the last "word" is very short (likely incomplete)
                        # Or if it's a single character/letter (likely cut off)
                        words = last_line.split()
                        if words:
                            last_word = words[-1]
                            # If last word is very short (1-2 chars) and not punctuation, likely incomplete
                            if len(last_word) <= 2 and last_word.isalpha():
                                # Remove the incomplete last word
                                lines[-1] = ' '.join(words[:-1]) if len(words) > 1 else ''
                            # If last word doesn't end with punctuation and line is short, might be incomplete
                            elif len(last_line) < 20 and not last_word.endswith(('.', '!', '?', ',', ';', ':')):
                                # Check if removing last word makes sense
                                # Only remove if it's clearly incomplete (very short word)
                                if len(last_word) < 4:
                                    lines[-1] = ' '.join(words[:-1]) if len(words) > 1 else ''
                    
                    # If after processing, last line is empty or just whitespace, remove it
                    if not lines[-1].strip():
                        lines = lines[:-1]
                
                # Reconstruct text
                generated_text = '\n'.join(lines)
                
                # Final check: if text doesn't end with punctuation and is not a speaker, 
                # try to find the last complete sentence
                # BUT: Be less aggressive - only remove if we have multiple sentences and last one is clearly incomplete
                if generated_text.strip():
                    # Find the last complete sentence (ends with . ! ?)
                    # Split by sentences
                    sentences = re.split(r'([.!?]+)', generated_text)
                    if len(sentences) > 3:  # Only if we have at least 2 complete sentences
                        # Reconstruct, keeping only complete sentences
                        complete_text = ''
                        for i in range(0, len(sentences) - 1, 2):
                            if i + 1 < len(sentences):
                                complete_text += sentences[i] + sentences[i + 1]
                        # If we have complete sentences, use them; otherwise keep original
                        if complete_text.strip():
                            # But check if we removed too much (more than 30% of text must remain)
                            # AND the last sentence must be very short (likely incomplete)
                            original_len = len(generated_text.strip())
                            complete_len = len(complete_text.strip())
                            if complete_len > original_len * 0.3:
                                # Check if last sentence in original is very short (likely incomplete)
                                last_sentence = sentences[-2] if len(sentences) >= 2 else ''
                                if len(last_sentence.strip()) < 15:  # Very short last sentence
                                    generated_text = complete_text.strip()
        
        return generated_text
    except Exception as e:
        import traceback
        return f"❌ Error during generation: {str(e)}\n\nPlease check:\n1. Model is uploaded to HuggingFace Model Hub\n2. Repository name is correct: shwethd/gpt2-shakespeare-124m\n3. File name is exactly: model_checkpoint_final.pt"


# Create Gradio interface
with gr.Blocks(title="GPT-2 124M Shakespeare Model") as demo:
    # Status indicator
    status_color = "🟢" if model_loaded else "🔴"
    status_text = "Model loaded successfully!" if model_loaded else "⚠️ Model not loaded - check HuggingFace Model Hub!"
    
    gr.Markdown(f"""
    # 🎭 GPT-2 124M Shakespeare Language Model
    
    {status_color} **Status:** {status_text}
    
    This is a 124M parameter decoder-only transformer model trained on Shakespeare's complete works.
    
    **Training Results:**
    - Final Loss: 0.095127 (Target: < 0.099999) ✅
    - Model Parameters: 124.44M
    - Training Steps: 1,637
    
    Enter a prompt below to generate Shakespeare-style text!
    
    {"⚠️ **Note:** If you see garbled/random text, the model may not have loaded correctly. Check the logs and ensure the model is uploaded to HuggingFace Model Hub: `shwethd/gpt2-shakespeare-124m`" if not model_loaded else ""}
    """)
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here (e.g., 'First Citizen:', 'ROMEO:', 'To be or not')",
                value="First Citizen:",
                lines=3
            )
            max_tokens = gr.Slider(
                label="Max Tokens",
                minimum=50,
                maximum=200,
                value=100,
                step=10
            )
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                info="Lower = more focused, Higher = more creative (0.7 recommended for better coherence)"
            )
            top_k = gr.Slider(
                label="Top-K",
                minimum=10,
                maximum=100,
                value=50,
                step=10,
                info="Number of top tokens to consider"
            )
            top_p = gr.Slider(
                label="Top-P (Nucleus)",
                minimum=0.1,
                maximum=1.0,
                value=0.85,
                step=0.05,
                info="Nucleus sampling - 0.85-0.9 recommended. Lower (0.3) = too restrictive, Higher (0.95+) = too random"
            )
            repetition_penalty = gr.Slider(
                label="Repetition Penalty",
                minimum=1.0,
                maximum=1.5,
                value=1.1,
                step=0.05,
                info="Penalize repeated tokens - higher = less repetition (1.1 recommended)"
            )
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Generated Text",
                lines=10,
                interactive=True,  # Make it interactive so users can select and copy
                show_copy_button=True  # Add copy button
            )
    
    # Example prompts with suggested parameters
    gr.Markdown("### Example Prompts (Click to try - includes optimal settings)")
    examples = gr.Examples(
        examples=[
            # Format: [prompt, max_tokens, temperature, top_k, top_p, repetition_penalty]
            ["First Citizen:", 100, 0.7, 50, 0.85, 1.1],
            ["ROMEO:", 100, 0.65, 45, 0.88, 1.15],  # Romantic - slightly lower temp
            ["To be or not", 80, 0.6, 40, 0.85, 1.2],  # Quote - more focused
            ["HAMLET:", 100, 0.7, 50, 0.85, 1.1],
            ["MACBETH:", 100, 0.7, 50, 0.85, 1.1],
            ["JULIET:", 100, 0.65, 45, 0.88, 1.15],  # Romantic
            ["KING:", 100, 0.7, 50, 0.85, 1.1],
            ["LADY MACBETH:", 100, 0.7, 50, 0.85, 1.1],
            ["OTHELLO:", 100, 0.7, 50, 0.85, 1.1],
            ["What light through yonder", 100, 0.65, 45, 0.88, 1.15],  # Romantic quote
            ["All the world's a stage", 100, 0.7, 50, 0.85, 1.1],  # Metaphorical
            ["Double, double toil and trouble", 80, 0.7, 50, 0.85, 1.15],  # Witches chant
            ["Friends, Romans, countrymen", 100, 0.7, 50, 0.85, 1.1],  # Speech
            ["A rose by any other name", 100, 0.65, 45, 0.88, 1.15],  # Romantic quote
        ],
        inputs=[prompt_input, max_tokens, temperature, top_k, top_p, repetition_penalty]
    )
    
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_tokens, temperature, top_k, top_p, repetition_penalty],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    **Note:** The model was trained on Shakespeare text and generates text in that style.
    Generated text may not always be coherent but should follow Shakespearean patterns.
    """)

if __name__ == "__main__":
    # Don't use share=True on HuggingFace Spaces
    demo.launch()


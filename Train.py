import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from datasets import load_dataset
import os

#stream data from Hugging Face and load Tiktoken
print("Connecting to Hugging Face and streaming FineWeb-Edu...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
data_iterator = iter(dataset)

print("Loading OpenAI Tiktoken...")
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab 

#hyperparameters
n_embd = 256       
n_layer = 6         
n_head = 6          
block_size = 128    
batch_size = 4      
grad_accum_steps = 16 
max_iters = 25000  

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#architecture definition
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

#dataloader and training loop
model = MicroGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) 

def get_batch():
    text_chunk = ""
    # Keep pulling until enough for a batch  
    while len(enc.encode(text_chunk)) < (block_size * batch_size) + 10:
        try:
            text_chunk += next(data_iterator)['text'] + "\n"
        except StopIteration:
            break
            
    tokens = enc.encode(text_chunk)
    data = torch.tensor(tokens, dtype=torch.long)
    
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

#training loop
print(f"Starting the Long Haul on {device}...")

for iter in range(max_iters):
    optimizer.zero_grad(set_to_none=True)
    
    for micro_step in range(grad_accum_steps):
        X, Y = get_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(X, Y)
            loss = loss / grad_accum_steps
        loss.backward()
        
    optimizer.step()

    if iter % 10 == 0:
        print(f"Iteration {iter}/{max_iters}: Loss = {loss.item() * grad_accum_steps:.4f}")

    # Save a safety checkpoint every 500 iterations
    if iter % 500 == 0 and iter > 0:
        torch.save(model.state_dict(), f'micro_gpt_{iter}.pt')
        print(f"--> Saved backup: micro_gpt_{iter}.pt")

print("\nTraining complete! Saving final model...")
torch.save(model.state_dict(), 'micro_gpt_final.pt')
print("Saved successfully as 'micro_gpt_final.pt'.")
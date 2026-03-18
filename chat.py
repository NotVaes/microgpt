import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import sys

#config
model_name = "microgpt.pt"

#hyperparameters (must match train.py)
n_embd = 256      
n_layer = 6       
n_head = 6          
block_size = 128     
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading Tiktoken...")
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab 

# the blueprint (must match train.py perfectly)
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

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        return logits

#load the  model
print(f"Loading model to {device}...")
model = MicroGPT().to(device)

try:
    model.load_state_dict(torch.load(f'{model_name}', map_location=device, weights_only=True))
    print("Brain successfully loaded! Type 'quit' to exit.")
except FileNotFoundError:
    print(f"ERROR: Could not find '{model_name}'. Run train.py first!")
    sys.exit()

model.eval()

#the chat loop
print("\n" + "="*40)
print(" MicroGPT Terminal Ready")
print("="*40 + "\n")

temperature = 0.8 
max_new_tokens = 75

while True:
    user_prompt = input("You: ")
    if user_prompt.lower() in ['quit', 'exit']:
        print("Shutting down...")
        break
        
    user_tokens = enc.encode(user_prompt)
    context = torch.tensor([user_tokens], dtype=torch.long, device=device)
    
    print("AI:  ", end="", flush=True)
    
    for _ in range(max_new_tokens):
        context_cropped = context[:, -block_size:]
        
        with torch.no_grad():
            logits = model(context_cropped)
            
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        context = torch.cat((context, idx_next), dim=1)
        
        next_word = enc.decode([idx_next.item()])
        print(next_word, end="", flush=True)
        
    print("\n")
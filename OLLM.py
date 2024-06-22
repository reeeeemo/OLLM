'''
    Created by Robert Oxley (reeeeemo on GitHub)
    
    This was created for learning how to create a LLM using PyTorch.
    
    I would reccomend looking into tensors as well, as they are crucial for understanding how PyTorch works. (n-dimensional arrays)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from sklearn.model_selection import train_test_split
import math

# Constants
embed_size = 256
num_heads = 8
hidden_dim = 512
num_layers = 6
num_epochs = 50
patience = 5
batch_size = 32


import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

'''
    Prepares dataset for training the model.
    
    Tokenizes input text and converts to numerical values (via a vocabulary)
'''
class TextDataset(Dataset):
    def __init__(self, text, vocab, tokenizer):
        self.text = text
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.tokens = [vocab[token] for token in tokenizer(text)]
       
    def __len__(self):
        return len(self.tokens) - 1
    
    def __getitem__(self, index):
        return torch.tensor(self.tokens[index:index+2])
    
'''
    Transformer Model
    
    Includes embedding, positional encoding, transformer encoder, and a linear layer.

    Embedding: Transforms input tokens into vectors of a fixed size
    Positional Encoding: Information about the position of tokens in the sequence
    Transformer Encoder: A stack of N encoder layers (a bunch of neural networks with multi-head, self-attention, and feedforward layers)
    Linear Layer: Maps output of transformer encoder to the size of the vocabulary
'''
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc(output)
        return output

''' Positional Encoding

    Adds information about the position of tokens in the sequence.
    
    Also adds dropout for regularization!
'''
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        
   
# Builds our vocabulary from the text, and our tokenizer
def build_vocab(text, tokenizer):
    counter = Counter(tokenizer(text)) 
    
    token2idx = {token: idx for idx, token in enumerate(counter.keys())}
    idx2token = {idx: token for token, idx in token2idx.items()}
    
    vocab = Vocab(counter)
    vocab.token2idx = token2idx
    vocab.idx2token = idx2token

    return vocab

# Training functions that handle evaluation and early stopping (based on validation loss)
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            src = data[:, 0].to(device).unsqueeze(1)
            tgt = data[:, 1].to(device)
            output = model(src).squeeze(1)
            loss = criterion(output.view(-1, output.size(-1)), tgt)
            total_loss += loss.item()
    return total_loss / len(loader)

def train_evaluate(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            src = data[:, 0].to(device)
            tgt = data[:, 1].to(device)
            optimizer.zero_grad()
            output = model(src.unsqueeze(0)).squeeze(0)
            loss = criterion(output.view(-1, output.size(-1)), tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader)}, Val Loss: {val_loss}')

def train_evaluate_early_stopping(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, patience):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            src = data[:, 0].to(device).unsqueeze(1)
            tgt = data[:, 1].to(device)
            optimizer.zero_grad()
            output = model(src).squeeze(1)
            loss = criterion(output.view(-1, output.size(-1)), tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader)}, Val Loss: {val_loss}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
        if patience_counter == patience:
            print("Early stopping triggered")
            model.load_state_dict(torch.load('best_model.pt'))
            break
     
def generate_text(model, vocab, tokenizer, text, max_len=50):
    model.eval()
    tokens = tokenizer(text)
    src = torch.tensor([vocab[token] for token in tokens]).unsqueeze(0).to(device)
    generated = src
    for _ in range(max_len):
        output = model(generated)
        next_token = torch.argmax(output[:, -1, :], dim=1)
        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        if next_token.item() == vocab['<eos>']:
            break
    generated_tokens = generated.squeeze().tolist()
    generated_text = [vocab.idx2token[token_id] for token_id in generated_tokens]
    return ' '.join(generated_text)

def main():
    # Tokenizer
    tokenizer = get_tokenizer('basic_english')
    text = open('hamlet.txt').read() # Text Dataset 
    
    # Build Vocabulary Corpus
    vocab = build_vocab(text, tokenizer)
    
    # Training / Validation Set Split
    lines = text.split('\n')
    train_text, val_text = train_test_split(lines, test_size=0.1)
    
    # Build Datasets
    train_dataset = TextDataset(' '.join(train_text), vocab, tokenizer)
    val_dataset = TextDataset(' '.join(val_text), vocab, tokenizer)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Init Model and optimizer / loss functions
    model = TransformerModel(vocab_size=len(vocab), embed_size=embed_size, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation
    train_evaluate_early_stopping(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=num_epochs, patience=patience)
    
    generative_text = "hello there bob"
    # Generation
    print(generate_text(model, vocab, tokenizer, generative_text, len(generative_text)))
    

if __name__== "__main__":
    main()




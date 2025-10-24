import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

# ---------------------------
# 1. Define the Model
# ---------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embeds)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out.squeeze(1)

# ---------------------------
# 2. Load Model & Vocab (cached)
# ---------------------------
@st.cache_resource
def load_model_and_vocab():
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    
    vocab_size = len(vocab)
    embed_dim = 100
    hidden_dim = 128
    output_size = 1
    num_layers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_size, num_layers).to(device)
    model.load_state_dict(torch.load("spam_classifier.pth", map_location=device))
    model.eval()
    
    return model, vocab, device

model, vocab, device = load_model_and_vocab()

# ---------------------------
# 3. Preprocessing
# ---------------------------
@st.cache_data
def encode(text, vocab, max_length=150):
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    token_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    
    # Padding & truncation
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids += [vocab['<pad>']] * (max_length - len(token_ids))
    
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

# ---------------------------
# 4. Streamlit UI
# ---------------------------
st.title("📩 SMS Spam Classifier")
st.write("Enter a message and check if it's Spam or Ham.")

user_input = st.text_area("Message:")

if st.button("Predict"):
    if user_input.strip():
        x = encode(user_input, vocab).to(device)
        with torch.no_grad():
            output = model(x)
            label = "Spam 🚨" if torch.sigmoid(output).item() > 0.5 else "Ham ✅"
            st.write(f"**Prediction:** {label}")
    else:
        st.warning("Please enter a message.")

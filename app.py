import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


st.write("# Next Text Generator")

st.sidebar.title("Model Info:")

st.sidebar.write("Made using a simple 2 hidden layered Neural Network, this text generator can predict next characters of the input text provided.")

st.sidebar.write("Here, the intension is not to generate meaningful sentences, we require a lot of compute for that. This app aims at showing how a vanilla neural network is also capable of capturing the format of English language, and generate words that are (very close to) valid words. Notice that the model uses capital letters (including capital I), punctuation marks and fullstops nearly correct. The text is generated paragraph wise, because the model learnt this from the text corpus.")

st.sidebar.write("This model was trained on a simple 4.52 MB text corpus titled: 'Shakespeare'")

no_of_chars = st.slider("Number of characters to be generated", 100, 2000, 1000)


# Open the file in read mode
with open('Shakespeare.txt', 'r') as file:
    # Read the entire content of the file
    thefile = file.read()

content = thefile[:-2000]
test = thefile[-2000:]


# Create a dictionary to store unique characters and their indices
stoi = {}
stoi['@'] = 0

# Iterate through each character in the string
i = 1
for char in sorted(content):
    # Check if the character is not already in the dictionary
    if char not in stoi:
        # Add the character to the dictionary with its index
        stoi[char] = i
        i+=1

itos = {value: key for key, value in stoi.items()}

def generate_text(model, inp, itos, stoi, block_size, max_len):

    context = [0] * block_size
    # inp = inp.lower()
    if len(inp) <= block_size:
      for i in range(len(inp)):
        context[i] = stoi[inp[i]]
    else:
      j = 0
      for i in range(len(inp)-block_size,len(inp)):
        context[j] = stoi[inp[i]]
        j+=1

    name = ''
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        # if ch == '.':
        #     break
        name += ch
        context = context[1:] + [ix]
    return name

# Function to simulate typing effect
def type_text(text):
    # Create an empty text element
    text_element = st.empty()
    s = ""
    for char in text:
        # Update the text element with the next character
        s += char
        text_element.write(s+'$ꕯ$')
        time.sleep(0.004)  # Adjust the sleep duration for the typing speed

    text_element.write(s)
    
class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size1, hidden_size2):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size1)
    self.lin2 = nn.Linear(hidden_size1, hidden_size2)
    self.lin3 = nn.Linear(hidden_size2, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x)) # Activation function : change this
    x = torch.sin(self.lin2(x))
    x = self.lin3(x)
    return x
  
# Embedding layer for the context

# emb_dim = 15
emb_dim = st.selectbox(
  'Select Embedding Size',
  (15,32), index=0)
emb = torch.nn.Embedding(len(stoi), emb_dim)

# block_size = 15
block_size = st.selectbox(
  'Select Block Size',
  (15,50), index=0)
emb = torch.nn.Embedding(len(stoi), emb_dim)
model = NextChar(block_size, len(stoi), emb_dim, 500, 300).to(device)
model = torch.compile(model)

# Activation Function
actFunc = st.selectbox(
  'Select Activation Function',
  ("sin", "relu", "tanh"), index=0)

inp = st.text_input("Enter text", placeholder="Enter valid English text.")

print(inp, itos, stoi, block_size, no_of_chars)
btn = st.button("Generate")
if btn:
    st.subheader("Seed Text")
    type_text(inp)
    model.load_state_dict(torch.load("Model_emb"+str(emb_dim)+"_block_size_"+str(block_size)+"_"+actFunc+".pth", map_location = device, weights_only=True))
    # model.load_state_dict(torch.load("gt_eng_model_upper_two_hid_layer_emb"+str(emb_dim)+"_block_size_"+str(block_size)+".pth", map_location = device))
    gen_txt = generate_text(model, inp, itos, stoi, block_size, no_of_chars)
    st.subheader("Generated Text")
    print(inp+gen_txt)
    type_text(inp+gen_txt)

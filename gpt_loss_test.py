# import torch
# import torch.nn.functional as F
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # Initialize GPT-2 model and tokenizer
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# # Set model to evaluation mode
# model.eval()

# # Input text
# input_text = "The quick brown"

# # Tokenize input text
# input_ids = tokenizer.encode(input_text, return_tensors="pt")

# # Generate next token probabilities
# with torch.no_grad():
#     outputs = model(input_ids=input_ids)
#     logits = outputs.logits[:, -1, :]  # Logits for the last token
#     next_token_probs = F.softmax(logits, dim=-1)

# # Sample the next token from the probability distribution
# next_token = torch.multinomial(next_token_probs, num_samples=1)

# # Compute the cross-entropy loss for the next token
# target_token_id = tokenizer.encode("fox", add_special_tokens=False)[0]  # Token ID for "fox"
# loss = F.cross_entropy(logits, torch.tensor([target_token_id]))

# breakpoint()
# # Print the sampled token and the loss
# print("Sampled Token:", tokenizer.decode(next_token.item()))
# print("Cross-Entropy Loss for Next Token:", loss.item())

import torch
import torch.nn.functional as F

# Initialize GPT-2 parameters (simplified for illustration)
num_tokens = 4  # Example vocabulary size
embedding_size = 768  # Example embedding size
hidden_size = 768  # Example hidden size
output_size = 10  # Example output size

# Example input token IDs
input_token_ids = torch.tensor([[101, 2052, 4248, 2829]],dtype=torch.float32)  # Example tokenized input text

# Initialize GPT-2 model parameters (simplified for illustration)
embedding_weights = torch.randn(embedding_size, num_tokens)
output_weights = torch.randn(num_tokens, hidden_size)
output_bias = torch.randn(num_tokens)

# Forward pass
embedding_output = torch.matmul(embedding_weights, input_token_ids.unsqueeze(-1)).squeeze(-1)
hidden_state = embedding_output  # Simplified, no attention layers

logits = torch.matmul(output_weights, hidden_state.T) + output_bias

# Softmax activation
next_token_probs = F.softmax(logits, dim=-1)

# Sample next token
import numpy as np
next_token_probs_numpy = next_token_probs.detach().numpy()
next_token_index = np.random.choice(num_tokens, p=next_token_probs_numpy.ravel())
next_token = torch.tensor(next_token_index).unsqueeze(0)

# Target token ID (for illustration purposes)
target_token_id = 1234  # Example target token ID

breakpoint()
# Compute cross-entropy loss
target_probs = torch.zeros((1, num_tokens))
target_probs[0, target_token_id] = 1  # One-hot encoding of target token
loss = F.cross_entropy(logits.unsqueeze(0), target_probs)

# Print sampled token and loss
print("Sampled Token:", next_token.item())
print("Cross-Entropy Loss for Next Token:", loss.item())

breakpoint()
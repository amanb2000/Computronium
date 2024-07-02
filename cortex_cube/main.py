# Main training loop for the cube cortex video predictive coderl:

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from cube_model import cube

# Set up the model
NUM_EPOCHS = 10

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps:0")
model = cube(kernel_size=3, num_channels_list=[16, 128, 16], num_blocks=3, block_overlap_depth=1, device=device).to(device)

# Set up the optimizer
model_input_convs = [conv.weight for conv in model.input_conv]
model_body_convs = [conv.weight for conv in model.body_conv]
model_params = model_input_convs + model_body_convs
optimizer = optim.Adam(model_params, lr=0.001)

# Set up data. For now we will use dummy data.

num_timesteps = 100
data = torch.tensor(np.random.rand(num_timesteps, 1, 3, 64, 64), dtype=torch.float32).to(device)

losses = []
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    loss = 0
    for t in range(num_timesteps-1):
        x = data[t]
        y = model(x)
        loss += model.loss(y[:, 0:3], data[t+1])
    
    loss.backward()
    optimizer.step()
    losses += [loss.item()]
    print("Epoch: {}, Loss: {}".format(epoch, loss.item()))


# Plot the loss over time
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
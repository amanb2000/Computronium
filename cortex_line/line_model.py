import torch
import torch.nn as nn
import torch.nn.functional as F

class line(nn.Module):
    def __init__(self, vocab_size, overlap_height=128, dims=[512, 1024, 512], num_blocks=1, dt=0.1):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_height = dims[0]
        self.overlap_height = overlap_height
        self.dt = dt

        self.wte = nn.Embedding(vocab_size, overlap_height)

        model_list = []
        for i in range(len(dims)-1):
            model_list.append(nn.Linear(dims[i], dims[i+1], bias=False))
            model_list.append(nn.Tanh())
        self.NUR = nn.Sequential(*model_list)

        self.ln_head = nn.Linear(overlap_height, vocab_size)
        self.ln_head.weight = self.wte.weight # share weights

        self.Phi = None
        self.Phi_height = (self.block_height - overlap_height) * num_blocks + overlap_height

    def forward(self, idx):
        x = self.wte(idx).squeeze(dim=1)
        B, C = x.shape
        assert C == self.overlap_height

        if self.Phi is None:
            self.Phi = torch.zeros(B, self.Phi_height)
        dPhidt = torch.cat((x, self.Phi[:, C:]), dim=1)
        
        dPhidt_block_list = [dPhidt[:, i*(self.block_height - self.overlap_height):(i+1)*(self.block_height - self.overlap_height)+self.overlap_height] for i in range(self.num_blocks)]
        
        dPhidt = torch.zeros_like(self.Phi)
        for i in range(self.num_blocks):
            dPhidt[:, i*(self.block_height - self.overlap_height):(i+1)*(self.block_height - self.overlap_height)+self.overlap_height] = self.NUR(dPhidt[:, i*(self.block_height - self.overlap_height):(i+1)*(self.block_height - self.overlap_height)+self.overlap_height] + self.NUR(dPhidt_block_list[i]))
        return dPhidt

    def loss(self, y, targets, weight_regularization=0.01, activation_regularization=0.01):
        logits = self.ln_head(y)
        ce = F.cross_entropy(logits, targets)

        # Weight regularization
        weight_reg = 0
        for layer in self.NUR:
            if isinstance(layer, nn.Linear):
                weight_reg = weight_reg + torch.norm(layer.weight)**2
        
        # Activation regularization for the shared layers only!
        activation_reg = 0
        if self.num_blocks > 1:
            shared_channel_idx = torch.tensor([(i+1)*(self.block_height - self.overlap_height) + j for i in range(self.num_blocks-1) for j in range(self.overlap_height)])
            activation_reg = torch.mean(self.Phi[:, shared_channel_idx])**2

        return ce + weight_regularization*weight_reg + activation_regularization*activation_reg

    @torch.no_grad
    def inference(self, idxs, num_tokens):
        self.Phi = None

        frames = []
        for idx in idxs[:-1]:
            dPhidt = self(torch.Tensor([idx]).to(dtype=torch.long))
            self.Phi = self.Phi + self.dt * dPhidt
            frames.append(idx)
        
        for _ in range(num_tokens):
            idx = frames[-1].unsqueeze(0)
            dPhidt = self(idx)
            self.Phi = self.Phi + self.dt * dPhidt
            frames.append(self.ln_head(self.Phi[:,:self.overlap_height]).argmax().unsqueeze(0))
        return frames


import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F


class cube(nn.Module):
    def __init__(self, 
                 kernel_size=3, 
                 length=64, 
                 num_channels_list=[16, 128, 16], 
                 num_blocks=3, 
                 block_overlap_depth=1, 
                 device="cpu", 
                 dt=0.1, 
                 leak=0, 
                 sparsity_frac=0.99):
        """
        args: 
            ...
            length: 
                augmentation to the height of the Phi tensor (must be larger 
                than the video frame).
        """
        super(cube, self).__init__()
        if num_channels_list[0] != num_channels_list[-1]:
            raise ValueError("First and last number of kernels must be the same")
        
        self.num_blocks = num_blocks
        self.block_overlap_depth = block_overlap_depth
        self.kernel_size = kernel_size
        self.num_channels_list = num_channels_list
        self.len_num_channels_list = len(num_channels_list)
        self.device = device

        self.input_conv = nn.ModuleList([nn.Conv2d(num_channels_list[i], num_channels_list[i+1], kernel_size=kernel_size, padding=kernel_size//2) for i in range(self.len_num_channels_list-1)])
        
        self.body_conv = nn.ModuleList([nn.Conv2d(num_channels_list[i], num_channels_list[i+1], kernel_size=kernel_size, padding=kernel_size//2) for i in range(self.len_num_channels_list-1)])

        self.block_length = num_channels_list[-1] - block_overlap_depth
        self.Phi_depth = self.num_blocks * self.block_length + block_overlap_depth

        self.Phi = [] 
        self.dt = dt
        self.length = length
        self.leak = leak
        self.sparsity_frac = sparsity_frac

    def forward(self, x_, instrument_correlations=True):
        """
        args: 
            x_: 
                input tensor of shape (N, 3, m, n)
            instrument_correlations:
                whether to compute the correlation between the shared channels
        returns:
            dPhidt: 
                tensor of shape (N, self.Phi_depth, m, n)
        """
        # make a mask with sparsity_frac of the values being 1
        mask = torch.rand_like(x_) > self.sparsity_frac
        x = x_ * mask

        batch_size = x.shape[0] # N, batch_size
        c = x.shape[1] # Number of channels
        assert c == 3, "Input tensor must have 3 channels (RGB)"
        
        assert self.length >= x.shape[2]
        m = self.length
        n = x.shape[3] # Height of video frame

        if len(self.Phi) == 0:
            self.Phi.append(torch.zeros(batch_size, self.Phi_depth, m, n).to(self.device))

        input_block_depth = self.num_channels_list[-1]

        # pad x
        x_padded = F.pad(x, (0,0,0, self.Phi[-1].shape[-2]-x.shape[-2]), mode='constant', value=0)

        dPhidt_input = torch.cat((x_padded, self.Phi[-1][:, 3:input_block_depth]), dim=1)

        for input_conv in self.input_conv:
            dPhidt_input = input_conv(dPhidt_input)
            dPhidt_input = torch.tanh(dPhidt_input)
            # maybe nonlinearity here

        # dPhidt_input should now be (N, input_block_depth, m, n)
        
        dPhidt_body_list = [self.Phi[-1][:, (i+1)*(self.block_length):(i+2)*(self.block_length)+self.block_overlap_depth] for i in range(self.num_blocks-1)]
        
        for block in range(self.num_blocks-1):
            for body_conv in self.body_conv:
                dPhidt_body_list[block] = body_conv(dPhidt_body_list[block])
                dPhidt_body_list[block] = torch.tanh(dPhidt_body_list[block])
                # maybe nonlinearity here
        
        # dPhidt_body_list should now be a list of length num_blocks-1, of tensors shape (N, block_length, m, n)

        dPhidt = torch.zeros_like(self.Phi[-1])
        dPhidt[:, 0:input_block_depth] += dPhidt_input
        for block in range(self.num_blocks-1):
            dPhidt[:, (block+1)*self.block_length:(block+2)*self.block_length+self.block_overlap_depth] += dPhidt_body_list[block]

        # Add 1 and divide by 2 for dPhidt[:, :3] 
        dPhidt[:, :3] = 0.5 * (dPhidt[:, :3] + 1)

        return dPhidt
    
    def loss(self, y, y_pred, weight_regularization=0.01, activation_regularization=0.01):
        batch_size = y.shape[0] # N, batch_size
        c = y.shape[1] # Number of channels
        m = y.shape[2] # Width of video frame
        n = y.shape[3] # Height of video frame

        MSE = torch.mean((y - y_pred[:, :, :m, :])**2)

        # Weight regularization
        weight_reg = 0
        for input_conv in self.input_conv:
            weight_reg += torch.norm(input_conv.weight)**2
        for body_conv in self.body_conv:
            weight_reg += torch.norm(body_conv.weight)**2
        
        # Activation regularization for the shared layers only!
        activation_reg = 0
        shared_channel_idx = torch.tensor([(i+1)*self.block_length + j for i in range(self.num_blocks-1) for j in range(self.block_overlap_depth)])
        #print("Check shared_channel_idx")
        activation_reg += torch.mean(self.Phi[-1][:, shared_channel_idx])**2

        return MSE + weight_regularization*weight_reg + activation_regularization*activation_reg

    def instrumentation_values(self):
        shared_channel_idx = torch.tensor([(i+1)*self.block_length + j for i in range(self.num_blocks-1) for j in range(self.block_overlap_depth)])
        return {
            "block_activations": [(i, torch.norm(self.Phi[-1][:, (i)*self.block_length:(i+1)*self.block_length+self.block_overlap_depth]).item()) for i in range(self.num_blocks)],
            "shared_channel_activations": [(i, torch.norm(self.Phi[-1][:, shared_channel_idx[i]]).item()) for i in range(self.num_blocks-1)],
        }
    
    def load_model(self, weight_file):
        # Load model weights
        state_dict = torch.load(weight_file, map_location=self.device)
        self.load_state_dict(state_dict)

    def save_model(self, weight_file):
        # Save model weights
        torch.save(self.state_dict(), weight_file)

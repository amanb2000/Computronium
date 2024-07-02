import torch
import torch.nn as nn

class cube(nn.Module):
    def __init__(self, kernel_size=3, num_channels_list=[16, 128, 16], num_blocks=3, block_overlap_depth=1):
        super(cube, self).__init__()
        if num_channels_list[0] != num_channels_list[-1]:
            raise ValueError("First and last number of kernels must be the same")
        
        self.num_blocks = num_blocks
        self.block_overlap_depth = block_overlap_depth
        self.kernel_size = kernel_size
        self.num_channels_list = num_channels_list
        self.len_num_channels_list = len(num_channels_list)

        self.input_conv = [nn.Conv2d(num_channels_list[i], num_channels_list[i+1], kernel_size=kernel_size, padding=kernel_size//2) for i in range(self.len_num_channels_list-1)]
        
        self.body_conv = [nn.Conv2d(num_channels_list[i], num_channels_list[i+1], kernel_size=kernel_size, padding=kernel_size//2) for i in range(self.len_num_channels_list-1)]

        self.block_length = num_channels_list[-1] - block_overlap_depth
        self.Phi_depth = self.num_blocks * self.block_length + block_overlap_depth

        self.Phi = None

    def forward(self, x):
        batch_size = x.shape[0]
        c = x.shape[1]
        assert c == 3, "Input tensor must have 3 channels (RGB)"
        m = x.shape[2]
        n = x.shape[3]

        if self.Phi is None:
            self.Phi = torch.zeros(batch_size, self.Phi_depth, m, n)

        input_block_depth = self.num_channels_list[-1]
        dPhidt_input = torch.cat((x, self.Phi[:, 3:input_block_depth]), dim=1)

        for input_conv in self.input_conv:
            dPhidt_input = input_conv(dPhidt_input)
            dPhidt_input = torch.tanh(dPhidt_input)
            # maybe nonlinearity here

        # dPhidt_input should now be (N, input_block_depth, m, n)
        
        dPhidt_body_list = [self.Phi[:, (i+1)*(self.block_length):(i+2)*(self.block_length)+self.block_overlap_depth] for i in range(self.num_blocks-1)]
        
        for block in range(self.num_blocks-1):
            for body_conv in self.body_conv:
                dPhidt_body_list[block] = body_conv(dPhidt_body_list[block])
                dPhidt_body_list[block] = torch.tanh(dPhidt_body_list[block])
                # maybe nonlinearity here
        
        # dPhidt_body_list should now be a list of length num_blocks-1, of tensors shape (N, block_length, m, n)

        dPhidt = torch.zeros_like(self.Phi)
        dPhidt[:, 0:input_block_depth] += dPhidt_input
        for block in range(self.num_blocks-1):
            dPhidt[:, (block+1)*self.block_length:(block+2)*self.block_length+self.block_overlap_depth] += dPhidt_body_list[block]

        return dPhidt


if __name__ == "__main__":

    # Test our model with an input of all zeros!

    model = cube()

    x = torch.zeros(1, 3, 32, 32)

    y = model(x)

    print(y)
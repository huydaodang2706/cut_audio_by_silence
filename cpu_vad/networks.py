import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
import torch.nn.functional as F
# Functions
##############################################################################
def get_network():
    return Vad_2021()


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# Networks
##############################################################################
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:tuple,
                 stride=(2,1), pad=0,
                 norm_fn='bn',
                 act='prelu'):
        super(Conv2dBlock, self).__init__()
        # pad = ((kernel_size[0] - 1) // 2 * dilation[0], (kernel_size[1] - 1) // 2 * dilation[1])
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=pad, bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm2d(out_channels))
        if act == 'relu':
            block.append(nn.ReLU())
        elif act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())
        elif act == 'tanh':
            block.append(nn.Tanh())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)



class Vad_2021(nn.Module):
    def __init__(self):
        super(Vad_2021, self).__init__()

        # self.encoder_video = self.make_video_branch(video_kernel_sizes, video_strides, nf=128, outf=256)
        self.layer_1 = Conv2dBlock(1, 16, (3,2))
        self.layer_2 = Conv2dBlock(16, 32, (3,2))
        self.layer_3 = Conv2dBlock(32, 64, (3,2))
        self.layer_4 = Conv2dBlock(64, 128, (3,2))
        self.padding = (1,0,1,1)
 
        self.gru = nn.GRU(input_size=512, hidden_size=256, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(512, 256),
                                nn.ReLU(True),
                                nn.Linear(256, 1))
        
    def forward(self, s):
        f_s = s
        print("input.shape:",f_s.shape)
        
        f_s = F.pad(f_s, self.padding, 'constant', 0)
        print("f_s.shape:",f_s.shape)
        f_s = self.layer_1(f_s)
        print("f_s.shape:",f_s.shape)
        print('--------')

        f_s = F.pad(f_s, self.padding, 'constant', 0)
        print("f_s.shape:",f_s.shape)
        f_s = self.layer_2(f_s)
        print("f_s.shape:",f_s.shape)
        print('--------')

        f_s = F.pad(f_s, self.padding, 'constant', 0)
        print("f_s.shape:",f_s.shape)
        f_s = self.layer_3(f_s)
        print("f_s.shape:",f_s.shape)
        print('--------')

        f_s = F.pad(f_s, self.padding, 'constant', 0)
        print("f_s.shape:",f_s.shape)
        f_s = self.layer_4(f_s)
        print("f_s.shape:",f_s.shape)
        print('--------')

        # Reshape tensor
        f_s = f_s.view(f_s.size(0), -1, f_s.size(3)) # (B, C1, T1)
        # print("f_s.shape:",f_s.shape)
        
        merge = f_s.permute(2, 0, 1)  # (T1, B, C1+C2)
        merge, _ = self.gru(merge)
        print("lstm.shape:",merge.shape)

        merge = merge.permute(1, 0, 2)# (B, T1, C1+C2)
        merge = self.fc1(merge)
        print("merge-shape: ", merge.shape )
        out = merge.squeeze(2)
        
        return out


def test():
    net = Vad_2021().to(torch.device('cpu'))
    # print(net)
    pytorch_total_params = sum(p.numel() for p in net.parameters())    
    print('Number of params: ', pytorch_total_params)
    if torch.cuda.device_count() >= 1:
        print('For single-GPU')
        # net = net.cuda()    # For single-GPU
    else:
        net = net
        print('Don\'t have gpu')
    s = torch.randn((1, 1, 60, 600))
    start = timeit.default_timer()
    out = net(s)
    stop = timeit.default_timer()
    print('Time: ', (stop - start))
    print(out.shape)
    # print(out)


if __name__ == '__main__':
    test()
    # s1 = torch.randn([10,10,10])
    # s2 = F.interpolate(s1, size=4)
    # print(s2.shape)
    # print(s1[:,:,:4])
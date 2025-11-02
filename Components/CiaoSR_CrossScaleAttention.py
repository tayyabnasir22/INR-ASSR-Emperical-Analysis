from Components.CiaoSR_BasicBlock import CiaoSR_BasicBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

class CiaoSR_CrossScaleAttention(nn.Module):
    def DefaultConv(self, in_channels, out_channels, kernel_size,stride=1, bias=True):
        return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)
    
    def SamePadding(self, images, ksizes, strides, rates):
        assert len(images.size()) == 4
        batch_size, channel, rows, cols = images.size()
        out_rows = (rows + strides[0] - 1) // strides[0]
        out_cols = (cols + strides[1] - 1) // strides[1]
        effective_k_row = (ksizes[0] - 1) * rates[0] + 1
        effective_k_col = (ksizes[1] - 1) * rates[1] + 1
        padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
        padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
        # Pad the input
        padding_top = int(padding_rows / 2.)
        padding_left = int(padding_cols / 2.)
        padding_bottom = padding_rows - padding_top
        padding_right = padding_cols - padding_left
        paddings = (padding_left, padding_right, padding_top, padding_bottom)
        images = torch.nn.ZeroPad2d(paddings)(images)
        return images


    def ReduceSum(self, x, axis=None, keepdim=False):
        if not axis:
            axis = range(len(x.shape))
        for i in sorted(axis, reverse=True):
            x = torch.sum(x, dim=i, keepdim=keepdim)
        return x

    def ExtractImagePatches(self, images, ksizes, strides, rates, padding='same'):
        assert len(images.size()) == 4
        assert padding in ['same', 'valid']
        batch_size, channel, height, width = images.size()

        if padding == 'same':
            images = self.SamePadding(images, ksizes, strides, rates)
        elif padding == 'valid':
            pass
        else:
            raise NotImplementedError('Unsupported padding type: {}.\
                    Only "same" or "valid" are supported.'.format(padding))

        unfold = torch.nn.Unfold(kernel_size=ksizes,
                                dilation=rates,
                                padding=0,
                                stride=strides)
        patches = unfold(images)
        return patches  # [N, C*k*k, L], L is the total number of such blocks

    def __init__(self, channel=64, reduction=2, ksize=3, scale=2, stride=1, softmax_scale=10, average=True):
        super(CiaoSR_CrossScaleAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.softmax_scale = softmax_scale

        self.scale = scale
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_1 = CiaoSR_BasicBlock(self.DefaultConv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())  
        self.conv_match_2 = CiaoSR_BasicBlock(self.DefaultConv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())  
        self.conv_assembly = CiaoSR_BasicBlock(self.DefaultConv, channel, channel, 1, bn=False, act=nn.PReLU())    
        #self.register_buffer('fuse_weight', fuse_weight)

        if 3 in scale:
            self.downx3 = nn.Conv2d(channel, channel, ksize, 3, 1)
        if 4 in scale:
            self.downx4 = nn.Conv2d(channel, channel, ksize, 4, 1)

        self.down = nn.Conv2d(channel, channel, ksize, 2, 1)    

    def forward(self, input):
        _, _, H, W = input.shape

        if not isinstance(self.scale, list):
            self.scale = [self.scale]

        res_y = []
        for s in self.scale:
            mod_pad_h, mod_pad_w = 0, 0
            if H % s != 0:
                mod_pad_h = s - H % s
            if W % s != 0:
                mod_pad_w = s - W % s
            input_pad = F.pad(input, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

            #get embedding
            embed_w = self.conv_assembly(input_pad)     # [16, 64, 48, 48]
            match_input = self.conv_match_1(input_pad)  # [16, 32, 48, 48]

            # b*c*h*w
            shape_input = list(embed_w.size())      # b*c*h*w
            input_groups = torch.split(match_input, 1, dim=0)  # 16x[1, 32, 48, 48]
            # kernel size on input for matching
            kernel = s * self.ksize

            # raw_w is extracted for reconstruction
            raw_w = self.ExtractImagePatches(embed_w, ksizes=[kernel, kernel],
                                        strides=[self.stride * s, self.stride * s],
                                        rates=[1, 1],
                                        padding='same') # [16, 2304, 576], 2304=64*6*6, 576=48*48/(2*2), [N, C*k*k, L] 

            # raw_shape: [N, C, k, k, L]
            raw_w = raw_w.view(shape_input[0], shape_input[1], kernel, kernel, -1) # [16, 64, 6, 6, 576]
            raw_w = raw_w.permute(0, 4, 1, 2, 3).contiguous()    # [16, 576, 64, 6, 6] raw_shape: [N, L, C, k, k]
            raw_w_groups = torch.split(raw_w, 1, dim=0)  # 16x[1, 576, 64, 6, 6]


            # downscaling X to form Y for cross-scale matching
            ref = F.interpolate(input_pad, scale_factor=1./s, mode='bilinear')  # [16, 64, 24, 24]
            ref = self.conv_match_2(ref)        # [16, 32, 24, 24]
            w = self.ExtractImagePatches(ref, ksizes=[self.ksize, self.ksize],
                                    strides=[self.stride, self.stride],
                                    rates=[1, 1],
                                    padding='same')   # [16, 288, 576], 288=32*3*3, 576=24*24
            shape_ref = ref.shape
            
            # w shape: [N, C, k, k, L]
            w = w.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1) # [16, 32, 3, 3, 576]
            w = w.permute(0, 4, 1, 2, 3).contiguous()    # [16, 576, 32, 3, 3] w shape: [N, L, C, k, k]
            w_groups = torch.split(w, 1, dim=0)     # 16x[1, 576, 32, 3, 3]

            y = []
            # 1*1*k*k
            #fuse_weight = self.fuse_weight

            for xi, wi, raw_wi in zip(input_groups, w_groups, raw_w_groups):
                # normalize
                wi = wi[0]  # [576, 32, 3, 3] [L, C, k, k]
                max_wi = torch.max(torch.sqrt(self.ReduceSum(torch.pow(wi, 2),
                                    axis=[1, 2, 3], keepdim=True)), self.escape_NaN) # 
                wi_normed = wi/ max_wi # 
                
                # Compute correlation map
                xi = self.SamePadding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # [1, 32, 50, 50]  xi: 1*c*H*W
                yi = F.conv2d(xi, wi_normed, stride=1)   # [1, 576, 48, 48] [1, L, H, W] L = shape_ref[2]*shape_ref[3]
                # yi = F.conv2d(xi.cpu(), wi_normed.cpu(), stride=1)  #TODO

                yi = yi.view(1, shape_ref[2] * shape_ref[3], shape_input[2], shape_input[3])  # [1, 576, 48, 48]  (B=1, C=32*32, H=32, W=32)
                # rescale matching score
                yi = F.softmax(yi*self.softmax_scale, dim=1)     # [1, 576, 48, 48]
                if self.average == False:
                    yi = (yi == yi.max(dim=1,keepdim=True)[0]).float()

                # deconv for reconsturction
                wi_center = raw_wi[0]   # [576, 64, 6, 6]
                yi = F.conv_transpose2d(yi, wi_center, stride=self.stride*s, padding=s)   #[1, 64, 96, 96]
                # yi = F.conv_transpose2d(yi, wi_center.cpu(), stride=self.stride*s, padding=s).cuda()  #TODO

                # add down
                if s == 2:
                    yi = self.down(yi)  #[1, 64, 48, 48]
                elif s == 3:
                    yi = self.downx3(yi)
                elif s == 4:
                    yi = self.downx4(yi)

                yi =yi/6.
                y.append(yi)

            y = torch.cat(y, dim=0)
            y = y[:, :, :H, :W]

            res_y.append(y)
        
        res_y = torch.cat(res_y, dim=1)

        return res_y
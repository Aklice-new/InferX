import torch
from torch import tensor


in_channels = 1
out_channels = 1
# weight = torch.randn(out_channels, in_channels, 3, 3)
weight = torch.arange(out_channels * in_channels * 3 * 3).reshape(out_channels, in_channels, 3, 3)
bias = torch.randn(out_channels)  
stride = [1, 1]
padding = [1, 1]
dilation = [1, 1]
groups = 1
kernel_h , kernel_w = weight.shape[2], weight.shape[3]
stride_h = stride[0]
stride_w = stride[1]
padding_h = padding[0]
padding_w = padding[1]
dilation_h = dilation[0]
dilation_w = dilation[1]

def img2col(input : tensor):
    print("input shape", input.shape)
    C, input_h, input_w = input.shape
    channel_size = input_h * input_w
    kernel_size = kernel_h * kernel_w
    out_h = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    print("out_h, out_w", out_h, out_w)
    img_col = torch.zeros(out_h * out_w, C * kernel_size).reshape(-1)
    input = input.reshape(-1)
    data_ptr = 0
    for c in range(C):
        for kr in range(kernel_h):
            for kc in range(kernel_w):
                input_row = -padding_h + kr * dilation_h
                for o_r in range(out_h):
                    if input_row < 0 or input_row >= input_h:
                        for o_c in range(out_w):
                            img_col[data_ptr] = 0
                            data_ptr += 1
                    else:
                        input_col = -padding_w + kc * dilation_w
                        for o_c in range(out_w):
                            if input_col < 0 or input_col >= input_w:
                                img_col[data_ptr] = 0
                                data_ptr += 1
                            else:
                                img_col[data_ptr] = input[c * channel_size + input_row * input_w + input_col]
                                data_ptr += 1
                            input_col += stride_w
                    input_row += stride_h
    img_col = img_col.reshape(out_h * out_w, in_channels * kernel_h * kernel_w)
    return img_col

def cpu_conv2d(input : tensor):
    N, C, H, W = input.shape
    weight_col = weight.reshape(out_channels, (in_channels // groups) * kernel_h * kernel_w)
    out_h = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    conv_out = torch.zeros(N, out_channels, out_h, out_w)
    for b in range(N):
        img_b = input[b]
        img_col = img2col(img_b)
        # img_col : [out_h * out_w, in_channels * kernel_h * kernel_w]
        # weight_col : [out_channels, in_channels * kernel_h * kernel_w]
        print("img_col shape", img_col.shape)
        print("weight_col shape", weight_col.shape)
        print("img_b", img_b)
        print("img_col", img_col)
        print("weight_col", weight_col)
        out = torch.matmul(weight_col, img_col.T) # + bias
        out = out.reshape(out_channels, out_h, out_w)
        conv_out[b] = out
    return conv_out
    

def torch_conv2d(input : tensor):
    return torch.conv2d(input=input, weight=weight, stride=stride, padding=padding, dilation=dilation, groups=groups)

def check(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    for i in range(a.shape[0]):
        if abs(a[i] - b[i]) > 1e-4:
            print("Error: {} != {}".format(a[i], b[i]))
            return
    assert torch.allclose(a, b, atol=1e-4), "Error: {} != {}".format(a, b)

if __name__ == '__main__':
    input = torch.arange(1 * in_channels * 9 * 9).reshape(1, in_channels , 9, 9)
    # input = torch.randn(1, in_channels, 9, 9)
    my_o = cpu_conv2d(input)
    torch_o = torch_conv2d(input)
    print("my_o shape", my_o.shape)
    print(my_o)
    check(my_o, torch_o)

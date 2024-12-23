import torch 

# Input shape : (1, 4, 5, 5)
input = torch.arange(100).reshape(1, 4, 5, 5).float()
# Conv2d
kernel_size = 3
stride = 1
padding = 1
dilation = 1
groups = 1 
input_channels = 4
output_channels = 4
# im2col
B, C, H, W = input.shape
KH, KW = kernel_size, kernel_size
SH, SW = stride, stride
PH, PW = padding, padding
DH, DW = dilation, dilation
OH = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1
OW = (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1
in_group_size = input_channels // groups
out_group_size = output_channels // groups
output = torch.zeros(B, output_channels, OH, OW).float()
# weights = torch.randn(output_channels, in_group_size, KH, KW).float()
weights = torch.zeros(output_channels, in_group_size, KH, KW).float()
# bias = torch.randn(output_channels).float()
bias = torch.zeros(output_channels).float()

for b in range(B):
    # im2col
    col = torch.zeros(OH * OW, input_channels * KH * KW)
    for g in range(groups):
        col_ = torch.zeros(OH * OW, in_group_size * KH * KW)
        for x in range(in_group_size):
            for oh in range(OH):
                for ow in range(OW):
                    for kh in range(KH):
                        for kw in range(KW):
                            ih = oh * SH - PH + kh * DH
                            iw = ow * SW - PW + kw * DW
                            if ih < 0 or ih >= H or iw < 0 or iw >= W:
                                continue
                            col_[oh * OW + ow, x * KH * KW + kh * KW + kw] = input[b, g * in_group_size + x, ih, iw]
        col[:, g * in_group_size * KH * KW : (g + 1) * in_group_size * KH * KW]  = col_
        # GEMM
        output_ = torch.zeros(out_group_size, OH , OW)
        for o in range(out_group_size):
            for i in range(OH * OW):
                output_[o, i // OW , i % OW] += (col_[i] @ weights[g * out_group_size + o].reshape(-1)) + bias[g * out_group_size + o]
        # print("fill interval : ", g * (out_group_size), (g + 1) * (out_group_size))
        output[b, g * (out_group_size) : (g + 1) * (out_group_size), :, :] = output_
# col = col.transpose(0, 1)
cols = [col]
print(col)
print("weight shape : ", weights.shape)
print("bias shape : ", bias.shape)

# Check with torch 
m = torch.nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
# print("m.weight shape : ", m.weight.shape)
# print("m.bias shape : ", m.bias.shape)

m.weight.data = weights
m.bias.data = bias
output_torch = m(input)
for i in range(output_channels):
    assert torch.allclose(output[:, i], output_torch[:, i], atol=1e-4), f"output {i} mismatched " 
print("All outputs are matched")
# print("torch : ")
# for i in range(OH):
#     print(output_torch[0, i, 0, 0])
# for i in range(OH):
#     print(output[0, i, 0, 0])
print("Input shape : ", input.shape)
print("Col shape : ", col.shape)
print("Output shape : ", output.shape)

# # print("Im2col output : ", col)

# print("Torch output : ", output_torch)
# print("My output : ", output)
with open('./test_data/input.txt', 'w') as file:
    for val in input.cpu().numpy().flatten():
        file.write(str(val) + ' ')

with open('./test_data/col' + str(1) + '.txt', 'w') as file:
        for val in cols[0].detach().cpu().numpy().flatten():
            file.write(str(val) + ' ')

with open('./test_data/output' + str(1) + '.txt', 'w') as file:
        for val in output.detach().cpu().numpy().flatten():
            file.write(str(val) + ' ')

with open('./test_data/weight'+ str(1) + '.txt', 'w') as file:
    for val in weights.cpu().numpy().flatten():
        file.write(str(val) + ' ')

with open('./test_data/bias' + str(1) + '.txt', 'w') as file:
    for val in bias.cpu().numpy().flatten():
        file.write(str(val) + ' ')




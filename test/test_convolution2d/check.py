import torch


# Input shape : (1, 64, 112, 112)
input = torch.randn(1, 64, 112, 112)
# Conv2d
m1 = torch.nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=3, stride=2, bias=True)
m2 = torch.nn.Conv2d(in_channels=64, out_channels= 64,kernel_size=3, stride=4, padding=1, bias=True, groups=64)
m3 = torch.nn.Conv2d(in_channels=64, out_channels= 128,kernel_size=5, stride=5, padding=1, bias=True)
m4 = torch.nn.Conv2d(in_channels=64, out_channels= 64,kernel_size=7, stride=3, padding=2, bias=True)

ms = [m1, m2, m3, m4]

output1 = m1(input)
output2 = m2(input)
output3 = m3(input)
output4 = m4(input)

outputs = [output1, output2, output3, output4]

print("Input shape : ", input.shape)
print("Output1 shape : ", output1.shape)
print("Output2 shape : ", output2.shape)
print("Output3 shape : ", output3.shape)
print("Output4 shape : ", output4.shape)
print("Weight1 shape : ", m1.weight.shape)
print("Weight2 shape : ", m2.weight.shape)
print("Weight3 shape : ", m3.weight.shape)
print("Weight4 shape : ", m4.weight.shape)
print("Bias1 shape : ", m1.bias.shape)
print("Bias2 shape : ", m2.bias.shape)
print("Bias3 shape : ", m3.bias.shape)
print("Bias4 shape : ", m4.bias.shape)


with open('./test_data/input.txt', 'w') as file:
    for val in input.cpu().numpy().flatten():
        file.write(str(val) + ' ')

for i in range(4):
    # weight
    with open('./test_data/weight' + str(i + 1) + '.txt', 'w') as file:
        for val in ms[i].weight.detach().cpu().numpy().flatten():
            file.write(str(val) + ' ')
    # bias
    with open('./test_data/bias' + str(i + 1) + '.txt', 'w') as file:
        for val in ms[i].bias.detach().cpu().numpy().flatten():
            file.write(str(val) + ' ')
    # output
    with open('./test_data/output' + str(i + 1) + '.txt', 'w') as file:
        for val in outputs[i].detach().cpu().numpy().flatten():
            file.write(str(val) + ' ')


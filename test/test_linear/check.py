import torch


# Input shape : (1, 512)
input = torch.randn(1, 512)
# Linear
m1 = torch.nn.Linear(in_features=512, out_features= 1000, bias=True)
m2 = torch.nn.Linear(in_features=512, out_features= 512, bias=True)
m3 = torch.nn.Linear(in_features=512, out_features= 256, bias=True)
m4 = torch.nn.Linear(in_features=512, out_features= 128, bias=True)

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
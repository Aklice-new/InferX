import torch


# Input shape : (1, 64, 112, 112)
input = torch.randn(1, 64, 112, 112)
# Conv2d
m1 = torch.nn.AdaptiveAvgPool2d((54,54))
m2 = torch.nn.AdaptiveAvgPool2d((37, 37))
m3 = torch.nn.AdaptiveAvgPool2d((1, 1))
m4 = torch.nn.AdaptiveAvgPool2d((99, 99))

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

with open('./test_data/input.txt', 'w') as file:
    for val in input.cpu().numpy().flatten():
        file.write(str(val) + ' ')

for i in range(4):
    # output
    with open('./test_data/output' + str(i + 1) + '.txt', 'w') as file:
        for val in outputs[i].detach().cpu().numpy().flatten():
            file.write(str(val) + ' ')


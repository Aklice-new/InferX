import torch


# Input shape : (1, 64, 112, 112)
input = torch.randn(1, 64, 112, 112)
# MaxPool2d
m1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
m2 = torch.nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
m3 = torch.nn.MaxPool2d(kernel_size=5, stride=5, padding=1)
m4 = torch.nn.MaxPool2d(kernel_size=7, stride=3, padding=2)

output1 = m1(input)
output2 = m2(input)
output3 = m3(input)
output4 = m4(input)

print("Input shape : ", input.shape)
print("Output1 shape : ", output1.shape)
print("Output2 shape : ", output2.shape)
print("Output3 shape : ", output3.shape)
print("Output4 shape : ", output4.shape)

with open('./test_data/input.txt', 'w') as file:
    for val in input.cpu().numpy().flatten():
        file.write(str(val) + ' ')

with open('./test_data/output1.txt', 'w') as file:
    for val in output1.cpu().numpy().flatten():
        file.write(str(val) + ' ')

with open('./test_data/output2.txt', 'w') as file:
    for val in output2.cpu().numpy().flatten():
        file.write(str(val) + ' ')

with open('./test_data/output3.txt', 'w') as file:
    for val in output3.cpu().numpy().flatten():
        file.write(str(val) + ' ')

with open('./test_data/output4.txt', 'w') as file:
    for val in output4.cpu().numpy().flatten():
        file.write(str(val) + ' ')



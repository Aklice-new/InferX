| 算子名称               | 状态                             | pytorch名称                                   | 备注       |
| ---------------------- | -------------------------------- | --------------------------------------------- | ---------- |
| 激活函数               |
| Relu                   | <input type="checkbox" checked>  | nn.ReLU                                       | 未经过测试 |
| Relu6                  | <input type="checkbox" checked>  | nn.ReLU6                                      | 未经过测试 |
| Sigmoid                | <input type="checkbox" checked>  | nn.Sigmoid                                    | 未经过测试 |
| Silu                   | <input type="checkbox" checked>  | nn.Silu                                       | 未经过测试 |
| HardSigmoid            | <input type="checkbox" checked>  | nn.Hardsigmoid                                | 未经过测试 |
| HardSwish              | <input type="checkbox" checked>  | nn.HardSwish                                  | 未经过测试 |
| TanH                   | <input type="checkbox" checked>  | nn.TanH                                       | 未经过测试 |
| cv                     |
| AdaptiveAveragePooling | <input type="checkbox"  checked> | nn.AdaptiveAvgPool2d,   F.adaptive_avg_pool2d | 未经过测试 |
| MaxPooling             | <input type="checkbox"  checked> | nn.MaxPool2d                                  | 未经过测试 |
| Batchnormal2D          | <input type="checkbox"  checked> | nn.BatchNorm2d                                | 未经过测试 |
| Upsample               | <input type="checkbox"  >        | nn.Upsample                                   | 未经过测试 |
| nn                     |
| cat                    | <input type="checkbox"  checked> | torch.cat                                     | 未经过测试 |
| flatten                | <input type="checkbox"  checked> | torch.flatten                                 | 未经过测试 |
| linear                 | <input type="checkbox"  checked> | nn.Linear                                     | 未经过测试 |
| view                   | <input type="checkbox"  >        | Tensor.view                                   | 未经过测试 |
| chunk                  | <input type="checkbox"  >        | Tensor.chunk                                  | 未经过测试 |
| split                  | <input type="checkbox"  >        | torch.split                                   | 未经过测试 |
| softmax                | <input type="checkbox"  >        | F.softmax                                     | 未经过测试 |
| transpose              | <input type="checkbox"  >        | torch.transpose                               | 未经过测试 |
| pnnx                   |
| expression             | <input type="checkbox"  checked> | pnnx.Expression                               | 未经过测试 |

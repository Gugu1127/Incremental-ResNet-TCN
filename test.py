import torch
from ResTCN import ResTCN
model = ResTCN().cuda()
inputs = torch.randn([16, 30, 3, 256, 256]).cuda()
outputs = model(inputs)
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

# 定义简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 定义自定义数据集
class RandomDataset(Dataset):
    def __init__(self, size=1000, input_size=10):
        self.data = torch.randn(size, input_size)
        self.labels = torch.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 训练过程
def train(rank, world_size):
    # 初始化分布式训练环境
    dist.init_process_group(
        backend='nccl',  # nccl 是 GPU 上的高效通信后端
        init_method='env://',  # 使用环境变量进行初始化
        world_size=world_size, 
        rank=rank
    )

    # 设置 GPU 设备
    torch.cuda.set_device(rank)
    model = SimpleModel().to(rank)
    
    # 将模型包装为 DDP
    model = DDP(model, device_ids=[rank])

    # 数据加载器
    dataset = RandomDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(10):  # 假设训练 10 个 epoch
        model.train()
        sampler.set_epoch(epoch)  # 每个 epoch 调用一次 sampler.set_epoch

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if rank == 0:  # 仅在 rank 0 上打印
            print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")

    # 关闭分布式训练环境
    dist.destroy_process_group()

# 测试过程
def test(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    model = SimpleModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # 假设有测试数据
    inputs = torch.randn(10, 10).to(rank)
    outputs = model(inputs)
    
    if rank == 0:  # 仅在 rank 0 上进行测试
        print("Test outputs: ", outputs)

    dist.destroy_process_group()

# 启动训练
def main():
    world_size = 2  # 使用 2 个 GPU 进行分布式训练

    # 启动多个进程进行训练
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

    # 测试
    # mp.spawn(test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torchsummary import summary

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# 读取数据并进行初步处理
# data = pd.read_csv('n8gap0.csv')
try:
    data = pd.read_csv('n8gap2.csv')
    print("文件读取成功！")
except Exception as e:
    print(f"读取文件时出错: {e}")
# # 添加5%随机误差
# noise_factor = 0.05
# noisy_labels = data['Rac_over_Rdc'] * (1 + noise_factor * np.random.randn(len(data)))

# 划分数据集
features = data[['Freq [MHz]', 'nlayer', 'lg(mm)', 'kdx', 'hma/hmcl']].values
labels = data['Rac_over_Rdc'].values
# 初始分割为训练+验证集和测试集
features_train_val, features_test, labels_train_val, labels_test= train_test_split(
    features, labels,  test_size=0.2, random_state=42
)

# 将训练+验证集分为训练集和验证集
features_train, features_valid, labels_train, labels_valid= train_test_split(
    features_train_val, labels_train_val, test_size=0.2, random_state=42)

# 转换为 torch tensors
features_train = torch.tensor(features_train, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.float32).view(-1, 1)
features_valid = torch.tensor(features_valid, dtype=torch.float32)
labels_valid = torch.tensor(labels_valid, dtype=torch.float32).view(-1, 1)
features_test = torch.tensor(features_test, dtype=torch.float32)
labels_test = torch.tensor(labels_test, dtype=torch.float32).view(-1, 1)


# 创建 TensorDataset
train_dataset = TensorDataset(features_train, labels_train )
valid_dataset = TensorDataset(features_valid, labels_valid )
test_dataset = TensorDataset(features_test, labels_test)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义神经网络结构
class Net(nn.Module):
    def __init__(self, num_features, num_hidden1, num_hidden2, num_hidden3, num_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, num_hidden3)
        self.fc4 = nn.Linear(num_hidden3, num_output)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# 初始化两个模型和优化器
net_mse = Net(num_features=5, num_hidden1=16, num_hidden2=32, num_hidden3=16, num_output=1)
criterion_mse = nn.MSELoss()
optimizer_mse = Adam(net_mse.parameters(), lr=0.001)
# 训练和评估函数
def train_and_evaluate(model, criterion, optimizer, train_loader, valid_loader,num_epochs=500):
    train_losses = []
    valid_losses = []
    # all_rmse = []  # 存储所有的预测误差

    for epoch in range(num_epochs):
        model.train()  # 开启训练模式
        train_loss = 0.0  # 初始化训练损失为0
        for data in train_loader:
            features, targets= data
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # 累加每个batch的损失

        train_loss /= len(train_loader)  # 计算平均训练损失
        train_losses.append(train_loss)

        # 验证步骤
        model.eval()  # 开启评估模式
        valid_loss = 0.0
        with torch.no_grad():
            errors = []  # 存储本epoch的预测误差
            for data in valid_loader:
                features, targets= data
                output = model(features)
                loss = criterion(output, targets)

                valid_loss += loss.item()
                # 计算RMSE的组成部分
                errors.extend((output - targets).view(-1).tolist())

            valid_loss /= len(valid_loader)
            valid_losses.append(valid_loss)

            # 计算这一轮的RMSE
            # rmse = np.sqrt(np.mean(np.square(errors)))
            # all_rmse.append(rmse)
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    return train_losses, valid_losses

def evaluate_on_test(model, test_loader):
    model.eval()
    total_loss = 0
    errors = []
    count = 0
    with torch.no_grad():
        for features, targets in test_loader:
            output = model(features)
            loss = (output - targets) ** 2
            total_loss += loss.sum().item()
            errors.extend(torch.abs(output - targets).view(-1).tolist())
            count += targets.size(0)
    rmse = np.sqrt(total_loss / count)
    return rmse, errors



train_losses_mse, valid_losses_mse = train_and_evaluate(net_mse, criterion_mse, optimizer_mse, train_loader, valid_loader)
test_rmse_mse, errors_mse = evaluate_on_test(net_mse, test_loader)


# 创建一个图形框架
plt.figure(figsize=(14, 7))

# 绘制训练损失曲线
plt.subplot(1, 2, 1)  # 1行2列，这是第一个图
plt.plot(train_losses_mse, label='MSE Loss - Train', color='red')
plt.plot(valid_losses_mse, label='MSE Loss - Valid', color='blue', linestyle='dashed')
plt.title('Comparison of Training Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


def plot_error_distribution(errors, title, filename,csv_filename):
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, color='blue', alpha=0.7, rwidth=0.85, density=True)

    # 计算平均误差和最大误差
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    plt.title(f"{title}\nMean Error: {mean_error:.2f}%, Max Error: {max_error:.2f}%")
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Density')
    plt.grid(True)

    # 显示计算的平均误差和最大误差
    plt.axvline(x=mean_error, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_error, plt.ylim()[1] * 0.95, f'Mean={mean_error:.2f}%', color='red')

    plt.axvline(x=max_error, color='green', linestyle='dashed', linewidth=1)
    plt.text(max_error, plt.ylim()[1] * 0.90, f'Max={max_error:.2f}%', color='green')

    plt.savefig(filename)  # 保存图像
    plt.close()  # 关闭图形
    error_data = pd.DataFrame({'Errors': errors})
    error_data.to_csv(csv_filename, index=False)
plot_error_distribution(errors_mse, 'Error Distribution for MSE Loss', 'mse_loss_distribution.png','maxave.csv')

total_params = count_parameters(net_mse)
print(f'Total trainable parameters: {total_params}')

torch.save(net_mse.state_dict(), 'A_n6.pth')
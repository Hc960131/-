import torch
import torch.nn as nn
import torchvision.models as models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class VisualAttention(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(VisualAttention, self).__init__()
        # 使用预训练的CNN提取图像特征
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size + embed_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)  # 输出注意力分数
        )

    def forward(self, images, hidden_state):
        # 提取图像特征 (batch_size, embed_size, H, W)
        features = self.cnn(images)
        batch_size, embed_size, H, W = features.shape

        # 将特征展平为序列 (batch_size, H*W, embed_size)
        features_flat = features.view(batch_size, -1, embed_size)

        # 计算注意力分数
        hidden_expanded = hidden_state.unsqueeze(1).expand(-1, H * W, -1)
        energy = self.attention(torch.cat([features_flat, hidden_expanded], dim=2))
        attention_scores = torch.softmax(energy, dim=1)  # (batch_size, H*W, 1)

        # 加权求和得到上下文向量
        context_vector = (attention_scores * features_flat).sum(dim=1)  # (batch_size, embed_size)

        return context_vector, attention_scores


# 使用示例
embed_size = 256
hidden_size = 512
model = VisualAttention(embed_size, hidden_size)
images = torch.randn(32, 3, 224, 224)  # 32张224x224的RGB图像
hidden = torch.randn(32, hidden_size)  # LSTM的隐藏状态
context, attention = model(images, hidden)
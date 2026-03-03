import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=16, img_size=28):
        super(VAE, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.channels = 1
        
        # 编码器（卷积网络）
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((3, 3))  # 强制输出3x3尺寸
        )
        
        # 计算编码器输出的特征维度
        feature_dim = 128 * 3 * 3  # 128x3x3=1152
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
        
        # 解码器（反卷积网络）
        self.decoder_fc = nn.Linear(latent_dim, feature_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.channels, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((img_size, img_size)),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """ Reparameterization trick """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        h = self.decoder_fc(z)
        h = h.view(-1, 128, 3, 3)  # 对应编码器的3x3输出
        recon_x = self.decoder(h)
        return recon_x
    
    def forward(self, x, labels=None):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, labels)
        return recon_x, mu, log_var

# class VAE(nn.Module):
#     """
#     This model is a VAE for MNIST, which contains an encoder and a decoder.
    
#     The encoder outputs mu_phi and log (sigma_phi)^2
#     The decoder outputs mu_theta
#     """
#     def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
#         """
#         Args:
#             - input_dim: the input image, 1 * 28 * 28 = 784
#             - hidden_dim: the dimension of the hidden layer of our MLP model
#             - latent_dim: dimension of hidden vector z
#         """
#         super(VAE, self).__init__()
        
#         # Encoder
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2_mu = nn.Linear(hidden_dim, latent_dim)  # mu_phi
#         self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)  # log (sigma_phi)^2

#         # Decoder
#         self.fc3 = nn.Linear(latent_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, input_dim)

#         self.act = nn.LeakyReLU(0.2)

#     def encode(self, x):
#         """ 
#         Encode the image into z, representing q_phi(z|x) 
        
#         Args:
#             - x: the input image, we have to flatten it to (batchsize, 784) before input

#         Output:
#             - mu_phi, log (sigma_phi)^2
#         """
#         h = self.act(self.fc1(x))
#         mu = self.fc2_mu(h)
#         log_var = self.fc2_logvar(h)
#         return mu, log_var

#     def reparameterize(self, mu, log_var):
#         """ Reparameterization trick """
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z, labels):
#         """ 
#         Decode z into image x

#         Args:
#             - z: hidden code 
#             - labels: the labels of the inputs, useless here
        
#         Hint: During training, z should be reparameterized! While during inference, just sample a z from random.
#         """
#         h = self.act(self.fc3(z))
#         recon_x =  torch.sigmoid(self.fc4(h))  # Using sigmoid to constrain the output to [0, 1]
#         return recon_x.view(-1, 28, 28)

#     def forward(self, x, labels):
#         """ x: shape (batchsize, 28, 28) labels are not used here"""
#         x = x.view(-1, 28 * 28)
#         mu, log_var = self.encode(x)
#         z = self.reparameterize(mu, log_var)
#         recon_x = self.decode(z, labels)
#         return recon_x, mu, log_var

# TODO: 2.3 Calculate vae loss using input and output
def vae_loss(recon_x, x, mu, log_var, var=0.5):
    """ 
    Compute the loss of VAE

    Args:
        - recon_x: output of the Decoder, shape [batch_size, 1, 28, 28]
        - x: original input image, shape [batch_size, 1, 28, 28]
        - mu: output of encoder, represents mu_phi, shape [batch_size, latent_dim]
        - log_var: output of encoder, represents log (sigma_phi)^2, shape [batch_size, latent_dim]
        - var: variance of the decoder output, here we can set it to be a hyperparameter
    """
    # TODO: 2.3 Finish code!
    # Reconstruction loss (MSE or other recon loss)
    # KL divergence loss
    # Hint: Remember to normalize of batches, we need to cal the loss among all batches and return the mean!
    batch_size = x.size(0)
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')/ (2 * var * batch_size)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1)
    kl_loss = torch.mean(kl_loss)

    # Total loss
    loss = recon_loss + kl_loss
    # raise ValueError("Not Implemented yet!")
    return loss

class GenModel(nn.Module):
    def __init__(self, latent_dim=16, img_size=28, num_classes=10):
        super(GenModel, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.channels = 1
        self.num_classes = num_classes
        
        # 编码器（卷积网络）
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels + 1, 32, kernel_size=3, stride=2, padding=1),  # 2个通道：图像+标签
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((3, 3))  # 强制输出3x3尺寸
        )
        
        # 计算编码器输出的特征维度
        feature_dim = 128 * 3 * 3  # 128x3x3=1152
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
        
        # 解码器（反卷积网络）
        self.decoder_fc = nn.Linear(latent_dim + num_classes, feature_dim)  # 输入增加标签维度
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.channels, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((img_size, img_size)),
            nn.Sigmoid()
        )
    
    def encode(self, x, labels):
        batch_size = x.size(0)
        
        # 打印输入信息用于调试
        # if self.training and x.size(0) == 512:  # 只在训练批次且批次大小为512时打印
        #     print(f"输入图像形状: {x.shape}, 类型: {x.dtype}")
        #     print(f"标签形状: {labels.shape}, 类型: {labels.dtype}, 值范围: [{labels.min()}, {labels.max()}]")
        
        # 确保标签是整数类型
        if labels.dtype != torch.long:
            labels = labels.long()
        
        # 将标签扩展为单通道图像
        labels_img = labels.view(-1, 1, 1, 1).expand(-1, 1, self.img_size, self.img_size).float() / self.num_classes
        
        # # 打印连接前的信息
        # if self.training and x.size(0) == 512:
        #     print(f"扩展后的标签图像形状: {labels_img.shape}, 类型: {labels_img.dtype}")
        
        # 将标签图像与输入图像连接
        x_with_labels = torch.cat([x, labels_img], dim=1)
        
        # 打印连接后的信息
        # if self.training and x.size(0) == 512:
        #     print(f"连接后的输入形状: {x_with_labels.shape}")
        
        h = self.encoder(x_with_labels)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """ Reparameterization trick """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        # 确保标签是整数类型
        if labels.dtype != torch.long:
            labels = labels.long()

        if torch.is_tensor(labels) and labels.dim() == 0:
            labels = labels.unsqueeze(0)  # 标量 -> [1]
    
        # 2. 处理一维/二维张量标签
        if labels.dim() == 2:
            if labels.size(1) == 1:
                labels = labels.squeeze(1)  # [batch_size, 1] -> [batch_size]
            else:
                raise ValueError(f"二维标签形状错误：应为[batch_size, 1]，实际形状 {labels.shape}")
        elif labels.dim() != 1:
            raise ValueError(f"标签维度错误：应为标量、一维或二维([batch_size, 1])，实际维度 {labels.dim()}")

            
        # 将标签转换为one-hot编码并与潜在向量连接
        labels_one_hot = torch.zeros(z.size(0), self.num_classes, device=z.device)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
        z_with_labels = torch.cat([z, labels_one_hot], dim=1)
        
        h = self.decoder_fc(z_with_labels)
        h = h.view(-1, 128, 3, 3)  # 对应编码器的3x3输出
        recon_x = self.decoder(h)
        return recon_x
    
    def forward(self, x, labels):
        # 确保标签是整数类型
        if labels.dtype != torch.long:
            labels = labels.long()
            
        mu, log_var = self.encode(x, labels)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, labels)
        return recon_x, mu, log_var

# # TODO: 3 Design the model to finish generation task using label
# class GenModel(nn.Module):
#     """
#     This model is a VAE for MNIST with label information, 
#     which contains an encoder and a decoder.
    
#     The encoder outputs mu_phi and log (sigma_phi)^2
#     The decoder outputs mu_theta
#     """
#     def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
#         """
#         Args:
#             - input_dim: the input image, 1 * 28 * 28 = 784
#             - hidden_dim: the dimension of the hidden layer of our MLP model
#             - latent_dim: dimension of hidden vector z
#             - num_classes: number of classes in the dataset (10 for MNIST)
#         """
#         super(GenModel, self).__init__()
        
#         # Encoder - 增加标签输入
#         self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)  # 输入维度增加标签
#         self.fc2_mu = nn.Linear(hidden_dim, latent_dim)  # mu_phi
#         self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)  # log (sigma_phi)^2

#         # Decoder - 增加标签输入
#         self.fc3 = nn.Linear(latent_dim + num_classes, hidden_dim)  # 输入维度增加标签
#         self.fc4 = nn.Linear(hidden_dim, input_dim)

#         self.act = nn.LeakyReLU(0.2)
#         self.num_classes = num_classes

#     def encode(self, x, labels):
#         """ 
#         Encode the image and labels into z, representing q_phi(z|x,y) 
        
#         Args:
#             - x: the input image, flattened to (batchsize, 784)
#             - labels: the labels of the inputs, shape (batchsize,)

#         Output:
#             - mu_phi, log (sigma_phi)^2
#         """

#         if labels.dim() == 2 and labels.size(1) == 1:
#             labels = labels.squeeze(1)  # 将 [batch_size, 1] 变为 [batch_size]
#         # 将标签转换为one-hot编码并与输入图像拼接
#         labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
#         x_with_labels = torch.cat([x, labels_onehot], dim=1)
        
#         h = self.act(self.fc1(x_with_labels))
#         mu = self.fc2_mu(h)
#         log_var = self.fc2_logvar(h)
#         return mu, log_var

#     def reparameterize(self, mu, log_var):
#         """ Reparameterization trick """
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z, labels):
#         """ 
#         Decode z and labels into image x

#         Args:
#             - z: hidden code 
#             - labels: the labels of the inputs, used to condition the generation
        
#         Output:
#             - reconstructed image, shape (batchsize, 28, 28)
#         """
#         if torch.is_tensor(labels) and labels.dim() == 0:
#             labels = labels.unsqueeze(0)  # 标量 -> [1]
    
#         # 2. 处理一维/二维张量标签
#         if labels.dim() == 2:
#             if labels.size(1) == 1:
#                 labels = labels.squeeze(1)  # [batch_size, 1] -> [batch_size]
#             else:
#                 raise ValueError(f"二维标签形状错误：应为[batch_size, 1]，实际形状 {labels.shape}")
#         elif labels.dim() != 1:
#             raise ValueError(f"标签维度错误：应为标量、一维或二维([batch_size, 1])，实际维度 {labels.dim()}")

#         # 将标签转换为one-hot编码并与隐变量拼接
#         labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
#         z_with_labels = torch.cat([z, labels_onehot], dim=1)
        
#         h = self.act(self.fc3(z_with_labels))
#         recon_x =  torch.sigmoid(self.fc4(h))  # Using sigmoid to constrain the output to [0, 1]
#         return recon_x.view(-1, 1, 28, 28)

#     def forward(self, x, labels):
#         """ 
#         x: shape (batchsize, 28, 28) 
#         labels: shape (batchsize,)
#         """
#         x = x.view(-1, 28 * 28)
#         mu, log_var = self.encode(x, labels)
#         z = self.reparameterize(mu, log_var)
#         recon_x = self.decode(z, labels)
#         return recon_x, mu, log_var
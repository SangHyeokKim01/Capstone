import torch
import torch.nn as nn

class ESC50_CNN_Ver1(nn.Module):
    def __init__(self, input_shape, num_classes, 
                        conv1_channels, conv2_channels, 
                        fc1_size, fc2_size, dropout):
        # super(ESC_CNN, self).__init__()
        super().__init__()
        input_channels = input_shape[0]
        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels = input_channels, out_channels=conv1_channels,
                kernel_size=(57, 6), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 3), stride=(1, 3)),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, 
                kernel_size=(1, 3),stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        # 더미 텐서를 통해 features 출력의 flatten dimension 계산
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)  
            flat_dim = out.view(1, -1).shape[1]  # (1, C_out*H_out*W_out)


        # 3) Classifier (Fully Connected layers)
        self.classifier = nn.Sequential(
            
            nn.Flatten(),
            nn.Linear(flat_dim, fc1_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fc2_size, num_classes)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        x = self.features(x)      # Conv2d + ReLU + MaxPool2d ...
        x = self.classifier(x)    # Flatten -> Linear -> ...
        return x


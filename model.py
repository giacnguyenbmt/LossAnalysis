import torch.nn as nn
import torch


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)


class LPRNetEnhanceV20(nn.Module):
    """
    V20: Upgrade from V16
        - restructure code
        - add padding for last maxpool layer, change the kernel size of the second layer of stage 2 to (1x5)
        - add a point wise conv layer after the last maxpool layer in order to replace maxpool3d to maxpool2d
        - use small basic block with depthwise conv layers
        - new stem with input shape[188, 24]

    New update:
        - use small_basic_block
        - change depthwiseconv to conv2d layers
        - remove redundant batchnorm2d layers
    """
    def __init__(self, class_num, dropout_rate,  **kwargs):
        super().__init__()
        self.class_num = class_num

        self._maxpool2d_1 = nn.MaxPool2d(3, stride=1)
        self._maxpool2d_2 = nn.MaxPool2d(3, stride=(1, 2))
        self._maxpool2d_3 = nn.MaxPool2d(3, stride=(1, 2), padding=(0, 1))
        self._stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=(1, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self._stage_1 = nn.Sequential(
            small_basic_block(ch_in=64, ch_out=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self._stage_2 = nn.Sequential(
            small_basic_block(ch_in=128, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            small_basic_block(ch_in=256, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self._stage_3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 5), stride=1), 
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU()
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        keep_features = list()

        x = self._stem(x)
        keep_features.append(x)
        x = self._maxpool2d_1(x)

        x = self._stage_1(x)
        keep_features.append(x)
        x = self._maxpool2d_2(x)

        x = self._stage_2(x)
        keep_features.append(x)
        x = self._maxpool2d_3(x)

        x = self._stage_3(x)
        keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(torch.mean(torch.mean(f_pow,axis = -1),axis = -1),axis = -1)
            f_mean = torch.unsqueeze(f_mean, 1)
            f_mean = torch.unsqueeze(f_mean, 1)
            f_mean = torch.unsqueeze(f_mean, 1)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        if torch.onnx.is_in_onnx_export():
            logits = logits.permute(0, 2, 1).softmax(2)

        return logits

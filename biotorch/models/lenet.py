class LeNet(nn.Module):
    def __init__(self, activation='tanh', num_classes=10):
        super(LeNet, self).__init__()

        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        # create layer operations
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, inputs):
        """
        forward pass, which is same for conventional feed-forward net
        :param inputs: inputs with shape [batch_size, in_features]
        :return: logit outputs from the network
        """
        out = self.conv1(inputs)
        out = self.activation(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.pool(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out

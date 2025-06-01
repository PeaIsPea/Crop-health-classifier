import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = self.make_block(int_chanels=3,out_chanels=8)
        self.conv2 = self.make_block(int_chanels=8, out_chanels=16)
        self.conv3 = self.make_block(int_chanels=16, out_chanels=32)


        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=6272, out_features=512),
            nn.LeakyReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes),

        )
    def make_block(self,int_chanels, out_chanels):
        return nn.Sequential(
            nn.Conv2d(in_channels=int_chanels, out_channels=out_chanels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_chanels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_chanels, out_channels=out_chanels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_chanels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )



    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)



        return x

if __name__ == '__main__':
    model = SimpleCNN()
    input_data = torch.rand(8, 3, 112, 112)
    if torch.cuda.is_available():
        model.cuda()
        input_data = input_data.cuda()
    while True:
        res = model(input_data)
        print(res.shape)
        break

#Simple Deep conv Q network that only outputs value
from imports import *


class DDQNSolver(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.calc_fc_shape(), 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


    #Helper function to calculate size of flattened array from conv layer shapes    
    def calc_fc_shape(self):
        x0 = torch.zeros(self.input_dim).unsqueeze(0)
        x0 = self.conv(x0)
        print ("Flattened layer size is", x0.flatten().shape[0])
        return x0.flatten().shape[0]
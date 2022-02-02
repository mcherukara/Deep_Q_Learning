#Dueling Deep Conv Net that outputs both value and advantage
from imports import *


class DDQNSolver(nn.Module):
    def __init__(self, output_dim, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        
        #Common conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim[0], out_channels=32, kernel_size=3, stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Flatten(),
        )
        
        #Fully connected advantage layers (output=no of actions)
        self.linear_adv = nn. Sequential(
            nn.Linear(self.calc_fc_shape(), 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        
        #Fully connected value layer, output is 1
        self.linear_val = nn.Sequential(
            nn.Linear(self.calc_fc_shape(),512),
            nn.ReLU(),
            nn.Linear(512,1)
        )
            
        #Copy whole network to target and turn off gradients
        self.target_conv = copy.deepcopy(self.conv)
        self.target_linear_adv = copy.deepcopy(self.linear_adv)
        self.target_linear_val = copy.deepcopy(self.linear_val)
        
        for p in self.target_conv.parameters():
            p.requires_grad = False
            
        for p in self.target_linear_adv.parameters():
            p.requires_grad = False
            
        for p in self.target_linear_val.parameters():
            p.requires_grad = False

            
    # Forward pass for online Dueling network
    def forward_online(self, input):
        x = self.conv(input)
        adv = self.linear_adv(x)
        val = self.linear_val(x)
        return val + adv - adv.mean()    


    #Forward pass for target Dueling network
    def forward_target(self,input):
        x = self.target_conv(input)
        adv = self.target_linear_adv(x)
        val = self.target_linear_val(x)
        return val + adv - adv.mean()
    

    #Run through the forward model depending on whether it is the online or offline network
    def forward(self, input, model="online"):
        if model == "online":
            return self.forward_online(input)
        elif model == "target":
            return self.forward_target(input)


    #Helper function to calculate size of flattened array from conv layer shapes    
    def calc_fc_shape(self):
        x0 = torch.zeros(self.input_dim).unsqueeze(0)
        x0 = self.conv(x0)
        print ("Flattened layer size is", x0.flatten().shape[0])
        return x0.flatten().shape[0]
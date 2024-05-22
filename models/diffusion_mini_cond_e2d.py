import torch
import torch.nn as nn
from diffusion_unet_parts import *

def initialize_weights(model):
    # Iterate over the model's parameters and initialize them
    for param in model.parameters():
        nn.init.normal_(param, mean=0, std=1)
    return model

def initialize_weights_xavier(model):
    # Iterate over the model's parameters and initialize them
    for param in model.parameters():
        nn.init.xavier_normal_(param)
    return model
    
def check_parameters(model):
    for param in model.parameters():
        if param.requires_grad:
            mean = torch.mean(param.data)
            std = torch.std(param.data)
            if mean != 0 or std != 1:
                return False

    return True

class UNET(nn.Module):
    def __init__(self,in_channels,interim_channels=32):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, height, width) -> (Batch_Size, interim_channels, height, width)
            SwitchSequentialC(nn.Conv2d(in_channels, interim_channels, kernel_size=3, padding=1)),
            
            # (Batch_Size, interim_channels, height, width) -> # (Batch_Size, interim_channels, height, width) -> (Batch_Size, interim_channels, height, width)
            SwitchSequentialC(UNET_ResidualBlock(interim_channels, interim_channels), UNET_AttentionBlock(4, 8), UNET_AttentionBlock3(4, 8, 256)),
            
            # (Batch_Size, interim_channels, height, width) -> # (Batch_Size, interim_channels, height, width) -> (Batch_Size, interim_channels, height, width)
            # SwitchSequentialC(UNET_ResidualBlock(interim_channels, interim_channels), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, interim_channels, height, width) -> (Batch_Size, interim_channels, Height / 2, Width / 2)
            SwitchSequentialC(nn.Conv2d(interim_channels, interim_channels, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, interim_channels, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 2, Width / 2)
            SwitchSequentialC(UNET_ResidualBlock(interim_channels, interim_channels*2),  UNET_AttentionBlock(4, 16),  UNET_AttentionBlock3(4, 16, 256)),
            
            # (Batch_Size, interim_channels*2, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 2, Width / 2)
            # SwitchSequentialC(UNET_ResidualBlock(interim_channels*2, interim_channels*2), UNET_AttentionBlock(4, 16)),
            
            # (Batch_Size, interim_channels*2, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 4, Width / 4)
            SwitchSequentialC(nn.Conv2d(interim_channels*2, interim_channels*2, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, interim_channels*2, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4)
            SwitchSequentialC(UNET_ResidualBlock(interim_channels*2, interim_channels*4),  UNET_AttentionBlock(4, 32), UNET_AttentionBlock3(4, 32, 256)),
            
            # (Batch_Size, interim_channels*4, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4)
            # SwitchSequentialC(UNET_ResidualBlock(interim_channels*4, interim_channels*4), UNET_AttentionBlock(4, 32)),
            
            # (Batch_Size, interim_channels*4, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height, Width)
            SwitchSequentialC(nn.Conv2d(interim_channels*4, interim_channels*4, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, interim_channels*4, Height, Width) -> (Batch_Size, interim_channels*4, Height, Width)
            SwitchSequentialC(UNET_ResidualBlock(interim_channels*4, interim_channels*4)),
            
            # (Batch_Size, interim_channels*4, Height, Width) -> (Batch_Size, interim_channels*4, Height, Width)
            # SwitchSequentialC(UNET_ResidualBlock(interim_channels*4, interim_channels*4)),
        ])

        self.bottleneck = SwitchSequentialC(
            # (Batch_Size, interim_channels*4, Height, Width) -> (Batch_Size, interim_channels*4, Height, Width)
            UNET_ResidualBlock(interim_channels*4, interim_channels*4), 
            
            # (Batch_Size, interim_channels*4, Height, Width) -> (Batch_Size, interim_channels*4, Height, Width)
            UNET_AttentionBlock(4, 32), 
            
            # (Batch_Size, interim_channels*4, Height, Width) -> (Batch_Size, interim_channels*4, Height, Width)
            UNET_ResidualBlock(interim_channels*4, interim_channels*4), 
        )
        
        self.decoders = nn.ModuleList([
            # (Batch_Size, interim_channels*8, Height, Width) -> (Batch_Size, interim_channels*4, Height, Width)
            SwitchSequentialC(UNET_ResidualBlock(interim_channels*8, interim_channels*4)),
            
            # (Batch_Size, interim_channels*8, Height, Width) -> (Batch_Size, interim_channels*4, Height, Width)
            # SwitchSequentialC(UNET_ResidualBlock(interim_channels*8, interim_channels*4)),
            
            # (Batch_Size, interim_channels*8, Height, Width) -> (Batch_Size, interim_channels*4, Height, Width) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4) 
            SwitchSequentialC(UNET_ResidualBlock(interim_channels*8, interim_channels*4), Upsample(interim_channels*4)),
            
            # (Batch_Size, interim_channels*8, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4)
            SwitchSequentialC(UNET_ResidualBlock(interim_channels*8, interim_channels*4), UNET_AttentionBlock(4, 32), UNET_AttentionBlock3(4, 32, 256)),
            
            # (Batch_Size, interim_channels*8, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4)
            # SwitchSequentialC(UNET_ResidualBlock(interim_channels*8, interim_channels*4), UNET_AttentionBlock(4, 32)),
            
            # (Batch_Size, interim_channels*6, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 2, Width / 2)
            SwitchSequentialC(UNET_ResidualBlock(interim_channels*6, interim_channels*4), UNET_AttentionBlock(4, 32), UNET_AttentionBlock3(4, 32, 256), Upsample(interim_channels*4)),
            
            # (Batch_Size, interim_channels*6, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 2, Width / 2)
            SwitchSequentialC(UNET_ResidualBlock(interim_channels*6, interim_channels*2), UNET_AttentionBlock3(4, 16, 256), UNET_AttentionBlock(4, 16)),
            
            # (Batch_Size, interim_channels*4, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 2, Width / 2)
            # SwitchSequentialC(UNET_ResidualBlock(interim_channels*4, interim_channels*2), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, interim_channels*3, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height, Width)
            SwitchSequentialC(UNET_ResidualBlock(interim_channels*3, interim_channels*2),UNET_AttentionBlock3(4, 16, 256), UNET_AttentionBlock(4, 16), Upsample(interim_channels*2)),
            
            # (Batch_Size, interim_channels*3, Height, Width) -> (Batch_Size, interim_channels, Height, Width) -> (Batch_Size, interim_channels, Height, Width)
            SwitchSequentialC(UNET_ResidualBlock(interim_channels*3, interim_channels),  UNET_AttentionBlock3(4, 8, 256),UNET_AttentionBlock(4, 8)),
            
            # (Batch_Size, interim_channels*2, Height, Width) -> (Batch_Size, interim_channels, Height, Width) -> (Batch_Size, interim_channels, Height, Width)
            SwitchSequentialC(UNET_ResidualBlock(interim_channels*2, interim_channels),  UNET_AttentionBlock(4, 8)),
            
            # (Batch_Size, interim_channels*2, Height, Width) -> (Batch_Size, interim_channels, Height, Width) -> (Batch_Size, interim_channels, Height, Width)
            # SwitchSequentialC(UNET_ResidualBlock(interim_channels*2, interim_channels), UNET_AttentionBlock(4, 16)),
        ])

    def forward(self, x, x_cond, context, time):
        # x: (Batch_Size, 4, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, interim_channels*4)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, x_cond, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, x_cond, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, x_cond, context, time)       
        return x


class DiffusionMiniCondD(nn.Module):
    def __init__(self,in_channels,interim_channels,out_channels,time_dim=320):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.unet = UNET(in_channels,interim_channels)
        self.final = UNET_OutputLayer(interim_channels, out_channels)
    
    def forward(self, latent, cond_latent, context, time):
        # latent: (Batch_Size, in_channels, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, interim_channels)

        # (1, interim_channels) -> (1, interim_channels*4)
        time = self.time_embedding(time)
        
        # (Batch, in_channels, Height , Width) -> (Batch, interim_channels, Height, Width)
        output = self.unet(latent, cond_latent, context, time)
        
        # (Batch, interim_channels, Height, Width) -> (Batch,out_channels, Height, Width)
        output = self.final(output)
        
        # (Batch, out_channels, Height, Width)
        return output




if __name__=="__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")  
    t = torch.rand((1,320), dtype=torch.float32, device=device)
    # time = time_embedding(t)
    context = torch.rand((1,1, 2, 100),device=device)
    x_cond = torch.rand((1,256),device=device)
    x = torch.rand((1, 1, 72, 72),device=device)
    model = DiffusionMiniCondD(in_channels=1,interim_channels=32,out_channels=1).to(device)
    # model = accelerator.prepare(model)
    torch.save(model.state_dict(), "model.pth")
    # print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: {:,}".format(num_params))   
    # output = model(x)
    # print(output.size())
    # model = ConvPart(2)
    print(model(x,x_cond,context,t).shape)

    # model = initialize_weights(model)

    # if check_parameters(model):
    #     print("All parameters are initialized with zero mean and unit std.")
    # else:
    #     print("Some parameters are not initialized with zero mean and unit std.")
    
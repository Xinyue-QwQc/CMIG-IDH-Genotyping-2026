from models.baseline.FoundBioNet.CMD_module import CMDModule
from models.baseline.FoundBioNet.DSF import DSFModule
from models.baseline.FoundBioNet.TAFE_module_New import BrainSegFounder
import torch.nn as nn 


class FoundBioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tafe = BrainSegFounder()
        self.dsf = DSFModule(input_dim_cmd=768,input_dim_tafe=128)
        self.cmd = CMDModule()
    def forward(self,x):
        x_tafe = self.tafe(x) # 2,768
        x_cmd = self.cmd(x) # 2, 128
        return self.dsf(x_tafe,x_cmd)
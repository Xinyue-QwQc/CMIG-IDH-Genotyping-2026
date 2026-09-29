import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from models.Revised_SAM_Med3D.encoder_for_clsV3_ import SAM_CLS
import torch 

# model = SAM_CLS(conv_adapter=True, ConvMoe=True, mismatch=True, text_embed=True,bg_know=False,in_channels=4)
# model = torch.nn.DataParallel(model).cuda()
model_check = torch.load("/home/cyx/Codes/SAM_MED3D_for_CLS/checkpoint/SAM_MED3D_Combined_New_split+No_ucsf_no_reduced_2025-04-15/model_epoch_best.pth")["state_dict"]

# msg = model.load_state_dict(model_check)
print()
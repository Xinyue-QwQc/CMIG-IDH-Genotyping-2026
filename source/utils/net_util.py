import numpy as np
import torch
from typing import  Optional

# 调整学习率
def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)


def tailor_and_concat(x, model,text_embed:Optional[torch.Tensor]=None,tg:Optional[torch.Tensor]=None):
    temp = []
    idh_temp=[]
    # 图像裁剪为八块
    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    for i in range(len(temp)):
        if text_embed is not None:
            out = model(temp[i],text_embed) #,tg
            if len(out) == 2:
                idh_out = out[0]
            else:
                idh_out = out
        else:
            idh_out = model(temp[i])
            if torch.any(torch.isnan(idh_out)):
                # idh_out = 0
                continue
        # grade_out = model['grade'](encoder_outs[3], encoder_outs[4])
        # print("idh_out:",idh_out)
        idh_temp.append(idh_out)

    idh_out = torch.mean(torch.stack(idh_temp), dim=0)
    # print("idh_out mean:",idh_out)
    return idh_out#,grade_out


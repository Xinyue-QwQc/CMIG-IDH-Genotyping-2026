import torch.distributed as dist



# 计算参数量
def cal_params(model):
    param_num = sum([params.nelement() for params in model.parameters()])
    return param_num/1e6

# 将损失分到其他GPU上，进行计算
def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor
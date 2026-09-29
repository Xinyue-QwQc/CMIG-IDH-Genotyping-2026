import math

def pr_permute(t, permute_ls):
    seq_len = t.shape[1]
    l = int(math.pow(seq_len, 1 / 3))

    value_ls = []

    # 对于每一个维度，计算其对应的值
    for i in range(len(permute_ls) - 1, -1, -1):  # permute_ls)-1,-1,-1
        value_ls.append(l ** i)

    permute_index = []

    for i in range(seq_len):
        # 相当于是获取三维坐标下的坐标  x y z (将序列重新映射为坐标)
        index_ls = [0] * len(permute_ls)
        for j in range(len(permute_ls)):
            if i >= value_ls[j]:
                index_ls[j] = i // value_ls[j]
                i = i % value_ls[j]

        # 得到重新映射的下标
        new_index = 0
        for k, v in zip(index_ls, permute_ls):
            new_index += k * value_ls[v]

        permute_index.append(new_index)

    # print(permute_index)

    t_permute = t[:, permute_index]

    return t_permute, permute_index


def reverse_permute(t, permute_index):
    ls = [0] * len(permute_index)
    for i, v in enumerate(permute_index):
        ls[v] = i

    t_reverse = t[:, ls]
    return t_reverse


def permute(t, permute_index):
    t_permute = t[:, permute_index]
    return t_permute

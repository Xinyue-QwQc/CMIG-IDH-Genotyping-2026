import os 
import json
# 读取json文件
def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

dic_confusion = read_json("result/SAM_MED3D_CLS_Multi_with_Lora_After_confusion/best_epoch_result.json")
dic_confusion_brain = read_json("result/SAM_MED3D_CLS_Multi_with_Lora_After_confusion_brain/best_epoch_result.json")

confusin_erro_case = dic_confusion["Error Case"]
confusin_erro_case_brain = dic_confusion_brain["Error Case"]

confusion_id = [dic["id"] for dic in confusin_erro_case]
confusion_id_brain = [dic["id"] for dic in confusin_erro_case_brain]

# 不在改进的方法中的error case就是改进方法学的更好的情况
# 在改进方法中的error case，不在为改进方法中的error case就是改进方法没有学到的地方
# 两者的交集就是两者都没有学好的地方，需要观察一下数据情况

intersection = list(set(confusion_id).intersection(set(confusion_id_brain)))

# 差集
# [2,3,4,5] - [1,2,3,4] = [5]
# [1,2,3,4] - [2,3,4,5] = [1]
difference1 = list(set(confusion_id).difference(set(confusion_id_brain)))  # 原始方法还没有学到的地方
difference2 = list(set(confusion_id_brain).difference(set(confusion_id)))  # 改进方法偏向性


#  输出结果
print("原始方法还没有学到的地方：")
for i in difference1:
    print(i)
print("改进方法偏向性：")
for i in difference2:
    print(i)
print("两者都没有学好的地方：")
print(len(intersection))
for i in intersection:
    print(i)
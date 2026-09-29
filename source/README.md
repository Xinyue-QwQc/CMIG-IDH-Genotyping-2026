# template
Standard codes for myself
template data
数据顺序(T1,T1ce,T2,Flair)
## 框架改进
1. ConvLora 使用ConvLora进行微调训练
2. ConvLora+Addition 对Block norm2和neck进行微调
3. ConvLora+Norm_tune 对Block的norm2进行微调
4. 前一个基础上进行Warmup，结果挺好，因为分布不同？
4. ConvLora+Moe 对Block的norm2进行微调 Warm
<br/>
利用类似SAM_US的方法来对Mismatch Signs进行提取，然后得到医学先验知识
利用已有的大语言模型作信息提示
## Branch Mis
该分支使用T2-Flair Mismatch信号来进行改进

## Downsample损失
Mismatch feature extract 的信息损失太多了，如何在增加信息量的同时，减少参数量。
(可以通过进行池化聚合所有信息，然后再使用1×1卷积)<br>
目前使用的Misfeature Extractor是repvit_m06,相比SEnet18，效果有所提升。<br>
在进行下采样之前，通过CABM关注相关Mismatch相关信息
<br>[Refer from zhihu](https://zhuanlan.zhihu.com/p/66520078)
## 最好模型
/home/cyx/Codes/SAM_MED3D_for_CLS/checkpoint/SAM_MED3D_Combined_Final_version+No_ucsf_no_reduced_2024-09-01/model_epoch_best.pth
<br>
/home/cyx/Codes/SAM_MED3D_for_CLS/checkpoint/SAM_MED3D_Combined_Ablation_adapter+No_ucsf_no_reduced_2024-09-02
实际上这个checkpoint是没有scconv的版本
## BrainSegFounder
best_weighted版本可用
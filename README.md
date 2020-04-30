# 新型冠状肺炎自监督分割任务

## 数据集
1. 20cases: https://zenodo.org/record/3757476#collapseOne 包含20个cases，有病灶、左肺、右肺的分割label（全部20个病例都用）
1. 国家生物信息中心: http://ncov-ai.big.ac.cn/download 同样是CT图像，但没有分割label（只用五个病例作为自监督的数据）

## 大概的任务流程
先用Unet去噪做自监督任务，再用Unet学到的特征做病灶分割任务。  
1. 数据处理： nii数据的可视化及格式转换、标准化（CT的窗前床位的分布）、分训练验证测试集（要按照病例分，同一个病例的数据不能出现在不同的数据集）
2. 自监督任务： 加噪声有两种（只加在肺部、整张图像都加）
3. 分割任务： 用上王欢的注意力模块、Diceloss
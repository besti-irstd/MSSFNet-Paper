# MSSFNet

**Mamba Spatial–Spectral Fusion Network**，面向红外小目标检测的空间—频率特征融合网络。

本仓库是按照论文模块名称整理的代码版本，包含模型、训练与测试入口、预测图生成、历史权重兼容工具及整理前后的一致性验证脚本。整理保留原有网络运算、参数初始化顺序和训练配置，并将模块命名统一为 **MSSFNet、AWFE、HWFP、MSDA、FCSM、CSA**。

仓库提供模型代码和验证脚本；**数据集、预训练权重和原始版本代码需另行准备**。当前没有新增训练结果或完整数据集性能报告。

# 项目目录

```text
MSSFNet-Paper/
├── README.md
├── requirements.txt
├── main.py                       # 训练与测试
├── generate.py                   # 预测图生成
├── model/
│   ├── MSSFNet.py                 # 整体网络
│   ├── loss.py                    # 损失函数
│   └── modules/
│       ├── awfe.py
│       ├── hwfp.py
│       ├── msda.py
│       ├── fcsm.py
│       ├── csa.py
│       └── basic.py               # 卷积、残差块与 Haar 小波算子
├── utils/
│   ├── checkpoint.py             # 权重读取与名称映射
│   ├── data.py                   # 数据集与预处理
│   ├── metric.py                 # 评价指标
│   └── tools.py
└── tools/
    └── verify_refactor.py        # 与原始版本进行一致性比较
```

训练时会在仓库内创建 `weight/MSSFNet-时间戳/`；预测结果默认保存到 `outputs/predictions/`。这些运行输出未包含在当前提交中。


# 测试

```bash
python main.py --mode test --dataset-dir "./data/IRSTD-1k" --weight-path "./pretrained/IRSTD-1K/weight.pkl" --base-size 256 --deep-supervision 0
```

测试入口输出 mIoU、Pd 和 Fa，其中 Fa 按原代码乘以 `1,000,000` 后显示。数据集名称不在代码中硬编码为可选项，其它数据集只要满足读取器布局即可传入；历史权重加载验证覆盖 IRSTD-1K、NUAA-SIRST、NUDT-SIRST。

# 训练

```bash
python main.py --mode train --dataset-dir "./data/IRSTD-1k" --batch-size 4 --epochs 800 --lr 0.05 --base-size 256 --crop-size 256 --deep-supervision 0
```


本仓库目前用于保存论文命名整理后的实现与验证脚本，论文指标应以相应实验记录为准。

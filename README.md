- todo list
- [x] 完成table cell box格式（box格式可以用来对单元格做检测，且可简化标注难度）转html算法
- [x] paddle表格识别代码和预训练模型转pytorch，预训练模型代码script/
- [x] 私有有线表格数据训练
    - [x] 训练、测试、inder代码重构
- [ ] 私有无线表格数据训练
- [ ] 表格识别模型cell location回归优化

# DATASETS
## Public：
[PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet): 与训练使用了这个数据集，可自行去官网下载。    
    
## Private：
私人数据暂不公开，后期会提供table cell box格式转HTML代码。

# train
    python train.py --config=configs/train_table_mv3.yml

# test
    python test.py --config=configs/test_table_mv3.yml

# infer
    python recognizer_table.py

# model
    百度基于Pubtabnet训练的模型（paddle转pytorch）：
        inference_model/juneli/finetune/table_rec/pd2pt.pt
    基于私人数据训练的有线表格识别模型（目前场景比较固定，后续会继续泛化，使用者也可自行finetune）：
        inference_model/juneli/finetune/table_rec/best.pt
# Pointer-Generation-Network-Chinese

## 数据预处理

准备如 `PreLCSTS` 目录下README文件所声明的数据，然后执行如下的代码完成数据的预处理，处理后的数据会生成在 `data` 目录下；

```shell
python make_data_files.py
```

通过编辑 `make_data_files.py` 中的preprocess函数，可以修改过滤和一些分句和过滤逻辑；

```python
def preprocess(x):
    x = str(x).replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '')
    regex = re.compile(r"[()+-.、；！：《》（）:——“”？_【】\/]")  # 保留逗号句号
    x = regex.sub('', x)
    mytext = jieba.cut(x, cut_all=False)
    return ' '.join(mytext)
```

## 模型训练

执行如下的命令（或者使用shell脚本，或者screen等等方式挂载）
```shell
python train.py
```

训练的checkpoint等会生成在 `PGN-Chinese/data/saved_models/` 路径下；

## 模型预测

### 整体测试

```shell
python eval.py --task test 

> 2022-03-31 14:18:58,698 - data_util.log - INFO - 0025000.tar rouge_1:0.3050 rouge_2:0.1560 rouge_l:0.2997
```

### 单条数据测试

具体的数据内容在eval.py里面修改就可以，**如果有具体数据场景需求的时候，可以选择多个checkpoint测试，然后选择一个最长的句子作为摘要生成结果**

```shell
python eval.py --task demo --load_model 0005000.tar
```

```
中国民用航空局航空安全办公室主任朱涛通报，本次事故飞机损毁严重，调查难度很大。鉴于调查工作刚刚开始，以目前掌握的信息，还无法对于事故的原因有一个清晰的判断。下一步调查组将全力以赴搜集各方证据，重点在事发现场飞行记录器的搜寻，并综合各方面信息开展事故原因分析工作，深入全面查明事故原因，一旦调查工作取得进展，将在第一时间公布。

> 2022-03-31 14:19:37,479 - data_util.log - INFO - 中国 首架 事故 飞机 损毁 严重 ， 调查
```
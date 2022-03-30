# BertSum

## 数据预处理

### Step 1 下载原始数据 

下载LCSTS2.0原始数据，下载途径。将 `LCSTS2.0/DATA` 目录下所有 **`PART_*.txt`** 文件放入 `BertSum-master_Chinese/raw_data` 

下载[bert-base-chiese](https://huggingface.co/bert-base-chinese/tree/main)的 `pytorch_model.bin`，保存在 `/data/sdb1/lyx/Text-Summarization/BertSum-Chinese` 目录下。

### Step 2 将原始文件转换成json文件存储

`BertSum-master_Chinese/src`目录下，运行：

```
python preprocess_LAI.py -mode format_raw -raw_path ../raw_data -save_path ../raw_data -log_file ../logs/preprocess.log
```

### Step 3 分句分词 & 分割文件 & 进一步简化格式

* 分句分词：首先按照符号['。', '！', '？']分句，若得到的句数少于2句，则用['，', '；']进一步分句

* 分割文件：训练集文件太大，分割成小文件便于后期训练。**分割后，每个文件包含不多于16000条记录**

`BertSum-master_Chinese/src` 目录下运行如下命令，会生成在 `../json_data/LCSTS` 目录下

```
python preprocess_LAI.py -mode format_to_lines -raw_path ../raw_data -save_path ../json_data/LCSTS -log_file ../logs/preprocess.log
```

注：这个过程相对耗时较长

### Step 4 句子标注 & 训练前预处理

* 句子预处理：找出与参考摘要最接近的n句话(相似程度以ROUGE衡量)，标注为1(属于摘要)

```
python preprocess_LAI.py -mode format_to_bert -raw_path ../json_data -save_path ../bert_data -oracle_mode greedy -n_cpus 2 -log_file ../logs/preprocess.log
```

会以.pt形式的文件生成在 `Text-Summarization/BertSum-Chinese/bert_data` 目录下

注：这个过程相对耗时较长

## 模型训练

注：第一次训练的时候可能会自动执行预训练模型下载等操作，现在把代码里面用到tokenizer的地方都改成了transformer(huggingface)的写法了；

在 `BertSum-master_Chinese/src` 目录下执行下列训练脚本，其中**三行代码区别是参数 -encoder设置了不同值(classifier & transformer & rnn)分别代表三种不同的摘要层**，这里还暂时没有全部调通；

### BERT+Classifier model

```shell
sh BertSum-Chinese/src/BERT-Classifier-train.sh
```

输出的checkpoint保存在 `BertSum-Chinese/models/bert_classifier` 路径下

### 模型提供了续接训练的能力

以BERT-Classifier为例，续接命令如下所示：

```shell
python train_LAI.py -mode train -encoder classifier -dropout 0.1 -bert_data_path ../bert_data/LCSTS -model_path ../models/bert_classifier -lr 2e-3 -visible_gpus 1 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 30000 -accum_count 2 -log_file ../logs/bert_classifier -use_interval true -warmup_steps 10000 \
../models/bert_classifier/model_step_xxxxx.pt 
```

## 模型评估/预测

模型训练完毕后，`BertSum-master_Chinese/src` 目录下，运行：

```shell
python train_LAI.py \
-mode test \
-bert_data_path ../bert_data/LCSTS \
-model_path MODEL_PATH \
-visible_gpus 1 \
-gpu_ranks 0 \
-batch_size 30000 \
-log_file LOG_FILE \
-result_path ../results/LCSTS \
-test_all \
-block_trigram False \
-test_from ../models/bert_transformer/model_step_30000.pt
```

- `MODEL_PATH` 是储存checkpoints的目录
- `RESULT_PATH` is where you want to put decoded summaries (default `../results/LCSTS`)

**中途自己环境遇到的bug解决**

pyrouge报错找不到settings.ini，在src目录下git clone一个新的ROUGE，然后设置pyrouge_set_rouge_path
```shell
(h1_abc_base) lyx@h5:/data/sdb1/lyx/Text-Summarization/BertSum-Chinese/src$ pyrouge_set_rouge_path /data/sdb1/lyx/Text-Summarization/BertSum-Chinese/src/pyrouge/tools/ROUGE-1.5.5
2022-03-29 21:26:45,158 [MainThread  ] [INFO ]  Set ROUGE home directory to /data/sdb1/lyx/Text-Summarization/BertSum-Chinese/src/pyrouge/tools/ROUGE-1.5.5.
(h1_abc_base) lyx@h5:/data/sdb1/lyx/Text-Summarization/BertSum-Chinese/src$ 
```

CalledProcessError: returned non-zero exit status - https://github.com/tagucci/pythonrouge/issues/4
```shell
cd pythonrouge/RELEASE-1.5.5/data/
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
```

BertConfig相关的一些错误：
- train_LAI.py中的训练和预测过程可能需要稍微调整下，训练用下边一行，测试用上边一行
  - from pytorch_pretrained_bert import BertConfig
  - from transformers import BertConfig
  
### for BERT+Classifier model

```shell
sh BERT-Classifier-eval.sh
```

输出结果在 `/data/sdb1/lyx/Text-Summarization/BertSum-Chinese/results/LCSTS_step10000.candidate`


## 使用自己的数据进行预测

数据示例见 ```BertSum-Chinese/raw_data```， 注意该步骤过程中的数据准备长度不能太短，否则无法抽取。

执行数据预处理步骤中的step3 step4，执行之前可以先清空之前的文件 `/data/sdb1/lyx/Text-Summarization/BertSum-Chinese/json_data` `/data/sdb1/lyx/Text-Summarization/BertSum-Chinese/bert_data`

重新生成后，只保留用来预测的数据，训练的数据可以暂时放在别的地方：
- `/data/sdb1/lyx/Text-Summarization/BertSum-Chinese/json_data/LCSTS.test.0.json`
- `/data/sdb1/lyx/Text-Summarization/BertSum-Chinese/bert_data/LCSTS.test.0.bert.pt`

执行如下脚本，注意脚本中的 `-report_rouge False` 参数，如果有gt的话可以设置为True
```shell
sh BERT-Classifier-eval.sh
```

输出结果在 `/data/sdb1/lyx/Text-Summarization/BertSum-Chinese/results/LCSTS_step10000.candidate`

![img.png](images/selfoutput.png)
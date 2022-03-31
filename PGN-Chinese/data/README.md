# data

The project structure of the folder is as follows:

```
└─data
    └─chunked/
    └─finished/
    └─unfinished/
    └─saved_models/
    ├─vocab.txt
```


The contents of these folders can be generated from the `PGN-Chinese/PreLCSTS` data by executing the following command in the root directory
```shell
python make_data_files.py
```
## 更新日志
### 4.11
#### 更新内容
· 增加focal-loss ，目前只适用于 nproc-per-node * per_device_train_batch_size % num_generation == 0的情况
· 使用focal-loss对v2.1进行训练，目前的参数：
    c = 0.031
    gamma = 2
    alpha = 0.056
· 将focal-loss的参数写进脚本

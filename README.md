# dnn_by_cnn_for_classifer

### 生成tfrecord文件

python conver_tfrecord.py --data_dir data/train/ --data_name train.tfrecord

python conver_tfrecord.py --data_dir data/validation/ --data_name val.tfrecord

将三个文件放在 data_set下 

### 训练模型

python train.py --data_dir data_set/ --model_dir save_models/ --tb_dir logs/ --batch_size 128 --set_name train.tfrecord --check_point model.ckpt

### 测试模型

python eval.py --data_dir data_set/ --model_dir save_models/ --batch_size 1200 --set_name val.tfrecord --check_point model.ckpt



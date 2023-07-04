# vaik-video-classification-pb-trainer
Train video classification pb model

## train_pb.py

### Usage

```shell
pip install -r requirements.txt
python train_pb.py --train_tfrecords_dir_path ~/.vaik-utc101-video-classification-dataset_tfrecords/train \
                --test_tfrecords_dir_path ~/.vaik-utc101-video-classification-dataset_tfrecords/test \
                --classes_txt_path ~/.vaik-utc101-video-classification-dataset_tfrecords/train/ucf101_labels.txt \
                --epochs 20 \
                --step_size 1000 \
                --frame_num 8 \
                --batch_size 16 \
                --image_height 224 \
                --image_width 224 \
                --skip_frame_ratio 1,2,4 \
                --test_sample_num 200 \
                --output_dir_path '~/.vaik-video-classification-pb-trainer/output_model'        
```

- train_tfrecords_dir_path & test_tfrecords_dir_path

```shell
.
├── test
│   ├── dataset.tfrecords-00000
│   ├── dataset.tfrecords-00001
・・・
│   └── ucf101_labels.txt
└── train
    ├── dataset.tfrecords-00000
    ├── dataset.tfrecords-00001
・・・
    └── ucf101_labels.txt

```

### Output

![vaik-video-classification-pb-trainer1](https://github.com/vaik-info/vaik-video-classification-pb-trainer/assets/116471878/07611900-8ed2-4979-8c45-56444ce98f86)

![vaik-video-classification-pb-trainer2](https://github.com/vaik-info/vaik-video-classification-pb-trainer/assets/116471878/0cf2a939-15d5-4f4f-997d-735fa540fb01)

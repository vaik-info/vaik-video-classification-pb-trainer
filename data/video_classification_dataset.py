import random
import glob
import os

import tqdm
import numpy as np
import tensorflow as tf


class VideoClassificationDataset:
    output_signature = None
    classes = None
    frame_num = None
    skip_frame_ratio = None
    input_size = None
    max_sample_num = None
    parsed_dataset = None

    def __new__(cls, tfrecords_dir_path, classes, name='ucf101', split='train', frame_num=16, skip_frame_ratio=(1, 2, 4, 8),
                input_size=(192, 192, 3), max_sample_num=None):
        cls.classes = classes
        cls.frame_num = frame_num
        cls.skip_frame_ratio = skip_frame_ratio
        cls.input_size = input_size
        cls.max_sample_num = max_sample_num
        cls.parsed_dataset = cls.__load_tfrecords(tfrecords_dir_path)
        cls.output_signature = (tf.TensorSpec(name=f'video', shape=(frame_num,) + input_size, dtype=tf.uint8),
                                tf.TensorSpec(name=f'class_index', shape=(), dtype=tf.int32))

        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=cls.output_signature
        )
        return dataset

    @classmethod
    def _generator(cls):
        step_index = 0
        while True:
            for parsed_data in cls.parsed_dataset:
                step_index += 1
                if cls.max_sample_num is not None and cls.max_sample_num < step_index:
                    return
                video, class_index = parsed_data
                video_array = np.zeros((cls.frame_num, ) + cls.input_size, dtype=np.uint8)
                random_skip_frame_ratio = random.choice(cls.skip_frame_ratio)
                video = video[::random_skip_frame_ratio]
                start_frame = random.randint(0, max(1, video.shape[0]-cls.frame_num))
                video = video[start_frame:min(start_frame+cls.frame_num, video.shape[0])]
                video = tf.image.resize_with_crop_or_pad(video, max(video.shape[1:3]), max(video.shape[1:3]))
                video = tf.image.resize(video, (cls.input_size[0], cls.input_size[1]))
                video_array[:video.shape[0], :, :, :] = video
                yield tf.cast(video_array, tf.uint8), tf.cast(class_index, tf.int32)

    @classmethod
    def __parse_tfrecord_fn(cls, example):
        feature_description = {
            'video': tf.io.FixedLenFeature([], tf.string),
            'shape': tf.io.FixedLenFeature([4], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(example, feature_description)
        video = tf.io.parse_tensor(example['video'], out_type=tf.uint8)
        shape = example['shape']
        video = tf.reshape(video, shape)
        class_index = example['label']
        return video, class_index

    @classmethod
    def __load_tfrecords(cls, tfrecords_dir_path):
        tfrecords_path_list = glob.glob(os.path.join(tfrecords_dir_path, '*.tfrecords-*'))
        raw_dataset = tf.data.TFRecordDataset(tfrecords_path_list)
        parsed_dataset = raw_dataset.map(cls.__parse_tfrecord_fn)
        return parsed_dataset


    @classmethod
    def get_all_data(cls, dataset):
        dataset = iter(dataset)
        data_list = []
        for data in tqdm.tqdm(dataset, desc='get_all_data', total=cls.max_sample_num):
            data_list.append(data)
        all_data_list = [None for _ in range(len(data_list[0]))]
        for index in range(len(data_list[0])):
            all_data_list[index] = tf.stack([data[index] for data in data_list])
        return tuple(all_data_list)
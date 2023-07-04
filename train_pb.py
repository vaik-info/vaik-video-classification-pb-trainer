import os
import argparse
from datetime import datetime
import pytz
import tensorflow as tf

from data import video_classification_dataset
from model import r2plus1d_lite
from callbacks import save_callback

def train(train_tfrecords_dir_path, test_tfrecords_dir_path, classes_txt_path, epochs, step_size, batch_size, frame_num, image_height, image_width, skip_frame_ratio,
          test_sample_num, output_dir_path):
    with open(classes_txt_path, 'r') as f:
        classes = f.readlines()
    classes = [label.strip() for label in classes]

    # train
    TrainDataset = type(f'TrainDataset', (video_classification_dataset.VideoClassificationDataset,), dict())
    train_dataset = TrainDataset(train_tfrecords_dir_path, classes, split='train', frame_num=frame_num, skip_frame_ratio=skip_frame_ratio,
                                 input_size=(image_height, image_width, 3))
    train_dataset = train_dataset.shuffle(128, reshuffle_each_iteration=True).padded_batch(batch_size=batch_size, padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.int32)))
    # valid
    ValidDataset = type(f'ValidDataset', (video_classification_dataset.VideoClassificationDataset,), dict())
    valid_dataset = ValidDataset(test_tfrecords_dir_path, classes, split='test', frame_num=frame_num, skip_frame_ratio=skip_frame_ratio,
                                 input_size=(image_height, image_width, 3), max_sample_num=test_sample_num)
    valid_data = video_classification_dataset.VideoClassificationDataset.get_all_data(valid_dataset)

    # prepare model
    model = r2plus1d_lite.prepare(frame_num, height=image_height, width=image_width, classes_num=len(classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy())

    # prepare callback
    save_model_dir_path = os.path.join(output_dir_path,
                                       f'{datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d-%H-%M-%S")}')
    prefix = f'step-{step_size}_batch-{batch_size}'
    callback = save_callback.SaveCallback(save_model_dir_path=save_model_dir_path, prefix=prefix)

    model.fit_generator(train_dataset, steps_per_epoch=step_size,
                        epochs=epochs,
                        validation_data=valid_data,
                        callbacks=[callback])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pb')
    parser.add_argument('--train_tfrecords_dir_path', type=str, default='~/.vaik-utc101-video-classification-dataset_tfrecords/train')
    parser.add_argument('--test_tfrecords_dir_path', type=str, default='~/.vaik-utc101-video-classification-dataset_tfrecords/test')
    parser.add_argument('--classes_txt_path', type=str, default='~/.vaik-utc101-video-classification-dataset_tfrecords/train/ucf101_labels.txt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--step_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--frame_num', type=int, default=16)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--skip_frame_ratio', nargs='+', type=int, default=(1, 2, 4))
    parser.add_argument('--test_sample_num', type=int, default=200)
    parser.add_argument('--output_dir_path', type=str, default='~/.video-classification-pb-trainer/output_model')
    args = parser.parse_args()

    args.train_tfrecords_dir_path = os.path.expanduser(args.train_tfrecords_dir_path)
    args.test_tfrecords_dir_path = os.path.expanduser(args.test_tfrecords_dir_path)
    args.classes_txt_path = os.path.expanduser(args.classes_txt_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    os.makedirs(args.output_dir_path, exist_ok=True)

    train(args.train_tfrecords_dir_path, args.test_tfrecords_dir_path, args.classes_txt_path, args.epochs, args.step_size,
          args.batch_size, args.frame_num, args.image_height, args.image_width, args.skip_frame_ratio,
          args.test_sample_num, args.output_dir_path)
import os
import pickle
import tensorflow as tf
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(input_files, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    for input_file in input_files:
        with open(input_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        for i in range(len(labels)):
            img_raw = images[i].tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(img_raw),
                'label': _int64_feature(labels[i])
            }))
            writer.write(example.SerializeToString())
    writer.close()

# Define paths
data_dir = 'cifar-10-batches-py'
train_files = [os.path.join(data_dir, f'data_batch_{i}') for i in range(1, 6)]
test_files = [os.path.join(data_dir, 'test_batch')]

os.makedirs('cifar10/train', exist_ok=True)
os.makedirs('cifar10/val', exist_ok=True)

print("Converting training data...")
convert_to_tfrecord(train_files, 'cifar10/train/train.tfrecords')
print("Converting validation data...")
convert_to_tfrecord(test_files, 'cifar10/val/test.tfrecords')
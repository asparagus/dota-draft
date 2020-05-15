import tensorflow as tf
from tensorflow import keras

BUCKET = None  # set from task.py
TRAIN_PATTERN = 'data/records/train/*.tfrecords'
EVAL_PATTERN = 'data/records/val/*.tfrecords'

# Define some hyperparameters
BATCH_SIZE = 512
TRAIN_EXAMPLES = 1000 * 1000
EVAL_INTERVAL = 300

LEARNING_RATE = 0.001

# CONSTANTS
NUM_HEROES = 130
NUM_PICKS = 10
FEATURES = {
    'radiant': tf.io.FixedLenFeature([5], dtype=tf.int64),
    'dire': tf.io.FixedLenFeature([5], dtype=tf.int64),
    'label': tf.io.FixedLenFeature([1], dtype=tf.int64),
}


def parse_tfrecord(serialized_example):
    example = tf.io.parse_single_example(serialized_example, FEATURES)
    x = tf.concat([example['radiant'], example['dire']], axis=0)
    y = example['label']

    return x, y


# load the training data
def load_dataset(pattern, batch_size=1, mode=tf.estimator.ModeKeys.EVAL):
    files = tf.data.Dataset.list_files(pattern)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.experimental.AUTOTUNE)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(10000).repeat()
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1) # take advantage of multi-threading; 1=AUTOTUNE
    return dataset


def model_fn(layers):
    picks = keras.layers.Input(shape=(NUM_PICKS, NUM_HEROES))
    midpoint = int(NUM_PICKS / 2)

    radiant_picks= [picks[:,i,:] for i in range(0, midpoint)]
    dire_picks = [picks[:,i,:] for i in range(midpoint, NUM_PICKS)]

    radiant = keras.layers.Add()(radiant_picks)
    dire = keras.layers.Add()(dire_picks)
    layer = keras.layers.Concatenate()([radiant, dire])

    for n in layers:
        layer = keras.layers.Dense(n, activation='relu')(layer)
    
    out = keras.layers.Dense(1, activation='sigmoid')(layer)
    model = keras.models.Model(inputs=picks, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=LEARNING_RATE))
    return model


def serving_input_fn():
    inputs = tf.placeholder(dtype=tf.int64, shape=[None, NUM_PICKS])
    features = inputs
    return tf.estimator.export.ServingInputReceiver(features, inputs)


def train_and_evaluate(output_dir):
    run_config = tf.estimator.RunConfig(save_checkpoints_secs=EVAL_INTERVAL, keep_checkpoint_max=3)
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model_fn([64, 32, 16, 1]),
        config=run_config,
    )

    train_file_path = 'gs://{}/{}'.format(BUCKET, TRAIN_PATTERN)
    eval_file_path = 'gs://{}/{}'.format(BUCKET, EVAL_PATTERN)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: load_dataset(train_file_path, BATCH_SIZE, mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=TRAIN_EXAMPLES,
    )
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: load_dataset(eval_file_path, 1000, mode=tf.estimator.ModeKeys.EVAL),
        steps=None,
        start_delay_secs=60,  # start evaluating after N seconds
        throttle_secs=EVAL_INTERVAL,  # evaluate every N seconds
        exporters=exporter,
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

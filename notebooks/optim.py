import os
from glob import glob
from json import dump
from gc import collect
from random import sample
from shutil import rmtree, copy
from argparse import ArgumentParser

import tensorflow as tf
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import hp
from hyperopt import STATUS_OK
from timeit import default_timer

RAW_IMAGE_HEIGHT = 224
RAW_IMAGE_WIDTH = 224
RAW_IMAGE_CHANNELS = 3

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_MOD_DIR = DATA_DIR + '_mod'
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

space = {
    'optim': hp.choice('optim', ['sgd', 'adamw']),
    'dropout': hp.uniform('dropout', 0.3, 0.6),
}


def main():
    parser = ArgumentParser()
    parser.add_argument('-n', dest='max_evals', type=int, default=100)
    args = parser.parse_args()

    # Data
    make_data_mod_uni(25, refresh=False)

    batch_size = 512
    global dataset_mod_train, dataset_valid
    dataset_mod_train = tf.keras.utils.image_dataset_from_directory(
        DATA_MOD_DIR,
        batch_size=batch_size,
        label_mode='categorical',
        image_size=(RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH),
        crop_to_aspect_ratio=True,
        #validation_split=0.95, #del
        #seed=1, #del
        #subset='training' #del
    )
    dataset_valid = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'valid'),
        batch_size=batch_size,
        label_mode='categorical',
        image_size=(RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH),
        crop_to_aspect_ratio=True
    )

    global iteration
    iteration = 0
    trials = Trials()
    best = fmin(
    fn=objective,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=args.max_evals
    )

    return trials


def get_classes():
    return {
                i: os.path.split(name)[-1] for i, name in
                enumerate(sorted(glob(os.path.join(DATA_DIR, 'test', '*'))))
           }

def make_data_mod_uni(files_per_class, refresh):
    if os.path.exists(DATA_MOD_DIR) and not refresh:
        return

    if os.path.exists(DATA_MOD_DIR):
        rmtree(DATA_MOD_DIR)

    for name in get_classes().values():
        os.makedirs(os.path.join(DATA_MOD_DIR, name))
        files_to_copy = glob(os.path.join(DATA_DIR, 'train', name, '*'))
        files_to_copy = sample(files_to_copy, min(files_per_class, len(files_to_copy)))
        for f in files_to_copy:
            copy(f, os.path.join(DATA_MOD_DIR, name))
    return

def objective(params):

    print(params)

    global iteration
    iteration += 1

    checkpoint_path = os.path.join(
        CHECKPOINTS_DIR,
        ('birds_checkpoint'
            + f'__optiter-{iteration}__'
            + '__'.join(f'{k}-{v}' for k, v in params.items())
        )
    )

    optims = {
        'sgd': tf.keras.optimizers.SGD(0.3),
        'adamw': tf.keras.optimizers.AdamW(0.0001)
    }
    optim = optims[params['optim']]
    dropout = params['dropout']

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.005,
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        min_delta=0.001,
        patience=2,
        cooldown=2,
        min_lr=1e-6
    )

    augment = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
    ])

    pretrained_model = tf.keras.applications.efficientnet.EfficientNetB0(
        input_shape=(RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH, RAW_IMAGE_CHANNELS),
        include_top=False,
        weights='imagenet',
        pooling='max'
    )
    pretrained_model.trainable = False

    inputs = pretrained_model.input
    x = augment(inputs)

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(525, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optim,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.F1Score(
                average='weighted', threshold=None, name='f1_score', dtype=None
            )
        ]
    )

    global dataset_mod_train, dataset_valid
    start = default_timer()
    train_history = model.fit(
        dataset_mod_train,
        steps_per_epoch=len(dataset_mod_train),
        validation_data=dataset_valid,
        validation_steps=len(dataset_valid),
        epochs=256,
        callbacks=[
            early_stopping,
            checkpoint_callback,
            reduce_lr
        ]
    )
    run_time = default_timer() - start
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    accuracy = train_history.history['accuracy']
    val_accuracy = train_history.history['val_accuracy']
    f1_score = train_history.history['f1_score']
    val_f1_score = train_history.history['val_f1_score']
    epochs = len(train_history.history['val_loss'])

    json_path = os.path.join(checkpoint_path, 'history.json')
    with open(json_path, 'w') as f:
        dump(
            {
                'params': params,'iteration': iteration,
                'loss': loss, 'val_loss': val_loss,
                'accuracy': accuracy, 'val_accuracy': val_accuracy,
                'f1_score': f1_score, 'val_f1_score': val_f1_score,
                'epochs': epochs,
                'train_time': run_time,
                'lr': [
                    str(lr) for lr in train_history.history['lr']
                ]
            },
            f,
            indent=4
        )

    tf.keras.backend.clear_session()
    del model
    collect()

    return {'loss': loss[-1],
            'status': STATUS_OK}

if __name__ == '__main__':
    trials = main()

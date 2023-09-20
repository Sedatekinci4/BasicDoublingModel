import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns

from keras import layers

np.set_printoptions(precision=3, suppress=True)

def print_ver():
    print(tf.__version__)


def create_input_data():
    input_data = np.linspace([1], [50], num=50)
    print("\nInput data has been created.\nThe input data is:\n")
    print(input_data)
    print(input_data.T)
    return input_data


def create_output_data(data):
    output_data = np.dot(data, 2)
    print("\nThe output data is:\n")
    print(output_data)
    return output_data


def prepare_dataset(input_raw, output_raw):
    column_names = ['Input', 'Output']
    array = np.concatenate((input_raw, output_raw), axis=1)
    print("\nArray concatenated..\n")
    print(array)
    result = pd.DataFrame(array, columns=column_names)
    print("\nAdded the column names for better look...\n")
    return result


def split_data(dataset_to_split):
    train_dataset = dataset_to_split.sample(frac=0.8, random_state=0)
    test_dataset = dataset_to_split.drop(train_dataset.index)
    return train_dataset, test_dataset


def separate_feature_label():
    # Separating the labels from features The label is our value to predict
    train_features = train_data.copy()
    test_features = test_data.copy()

    train_labels = train_features.pop('Output')
    test_labels = test_features.pop('Output')

    return train_features, train_labels, test_features, test_labels


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [output]')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_guess(p_x,p_y):
    plt.scatter(train_features['Input'], train_labels, label='Data')
    plt.plot(p_x, p_y, color='k', label='Predictions')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    print_ver()

    input = create_input_data()
    output = create_output_data(input)

    dataset = prepare_dataset(input, output)
    print("\nDataset has been created....\n")
    print(dataset)


    train_data, test_data = split_data(dataset)
    print(train_data)
    print(test_data)

    # Inspecting our train set
    print(train_data.describe().transpose())

    train_features, train_labels, test_features, test_labels = separate_feature_label()

    # Label = expected output
    # Feature = input given to the model
    print("\nTrain features are:\n")
    print(train_features)

    print("\nTrain labels are:\n")
    print(train_labels)

    print("\nTest features are:\n")
    print(test_features)

    print("\nTest labels are:\n")
    print(test_labels)

    print(train_data.describe().transpose()[['mean', 'std']])

    # We use normalizer because our data has different scales and ranges.
    # Features are multiplied by the model weights, so we should normalize to see better results
    # Creating the layer
    normalizer = tf.keras.layers.Normalization(axis=-1)
    # for fitting the state of preprocessing layer
    normalizer.adapt(np.array(train_features))
    print(normalizer.mean.numpy())

    print("First")
    first = np.array(train_features[:1])
    print(train_features)

    with np.printoptions(precision=2, suppress=True):
        print('First example:', first)
        print()
        print('Normalized:', normalizer(first).numpy())

    # Linear Regression
    # (We will try to predict output from input
    guess = np.array(train_features['Input'])

    # We created an numpy array made of the Horsepower values. Then Normalize it and fit the horsepower datas
    guess_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
    guess_normalizer.adapt(guess)

    # Build the keras sequential model
    linear_model = tf.keras.Sequential([
        guess_normalizer,
        layers.Dense(units=1)
    ])

    # Prints the summary of model
    print(linear_model.summary())

    # Prints the untrained model on first 10 Horsepower values.
    print(linear_model.predict(guess[:10]))

    # Configuring training procedure (LOOK LATER FOR SURE)
    linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    # we use keras model.fit to do the training for 100 epochs
    # Epoch means: one complete pass of the training dataset through the algorithm
    history = linear_model.fit(
        train_features['Input'],
        train_labels,
        epochs=300,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split=0.1)

    # Printing the models training process for last 5 elements
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    # Plot the model graphically
    plot_loss(history)

    # Collecting the results to use later
    test_results = {}

    test_results['linear_model'] = linear_model.evaluate(
        test_features['Input'],
        test_labels, verbose=0)

    # See how our model predicts
    x = tf.linspace(0.0, 250, 251)
    y = linear_model.predict(x)

    z = linear_model.predict([450.5])
    print(z)
    plot_guess(x, y)

    # test_results['linear_model'] = linear_model.evaluate(
    #     test_features, test_labels, verbose=0)

    print(pd.DataFrame(test_results, index=['Output Error']).T)
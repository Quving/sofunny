import os

from keras.engine.saving import model_from_json


def export_model(model, modelname):
    """
    Export model to local directory.
    Args:
        model:
        filename:

    Returns:

    """
    directory = 'models'
    filename = os.path.join(directory, modelname)

    print("Persist model completely in '{}'.".format(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open('{}.json'.format(filename), 'w') as outfile:
        outfile.write(model.to_json(sort_keys=True, indent=4, separators=(',', ': ')))

    # Save weights
    model.save('{}.h5'.format(filename))


def load_model_from_file(modelname):
    """
    Local model from local directory.
    Args:
        filename:

    Returns:

    """
    directory = 'models'
    filename = os.path.join(directory, modelname)

    with open(filename + '.json', 'r') as file:
        model = model_from_json(file.read())
        file.close()

    model.load_weights(filename + '.h5')
    return model

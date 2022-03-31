from model import MaskRGNetwork
from dataset import PushAndGraspDataset
import yaml
import numpy as np
import os
from torch.utils import data as td


def main():

    with open('model_config.yaml') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    # create the model
    model = MaskRGNetwork(configuration)

    # create dataset objects and set the model
    dataset = PushAndGraspDataset(configuration)
    test_indices = os.path.join(configuration['dataset']['path'], configuration['dataset']['test_indices'])
    test_subset = td.Subset(dataset, test_indices)
    # Training:
    # model.set_data(dataset)
    # model.train_model()

    # Evaluation:
    # this loads the saved weights from the file in the config file
    model.load_weights()
    # load a new dataset for the evaluation
    model.set_data(test_subset, is_test=True, batch_size=20)

    # evaluate
    res = model.evaluate_model()
    np.save('results.npy', res)
    with open('results.txt', 'w') as output:
        output.write(res)


if __name__ == "__main__":
    main()
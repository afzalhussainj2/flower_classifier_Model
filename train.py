import argparse
from train_fnc import loading_data, train_model


parser = argparse.ArgumentParser()

#postional arguments
parser.add_argument('data_dir', type='train-fnc.py/.data_loading', help='Data directory')

#optional arguments
parser.add_argument('-save_dir', type=str, help='Save directory')
parser.add_argument('-arch', type=str, help='Model architecture')
parser.add_argument('-learning_rate', type=float, help='Learning rate')
parser.add_argument('-hidden_units', type=int, help='Hidden units')
parser.add_argument('-epochs', type=int, help='Number of epochs')
parser.add_argument('-gpu', type=str, help='Use GPU if available')

args = parser.parse_args()

data_loader = loading_data(args.data_dir)
train_model(data_loader, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
import argparse
from predict_fnc import predict, process_image, load_checkpoint

parser = argparse.ArgumentParser()

#postional arguments
parser.add_argument('img', type=str, help='Image file to predict')
parser.add_argument('checkpoint', type=str, help='Model checkpoint to load')

#optional arguments
parser.add_argument('--top_k', type=int, default=1, help='Top K most likely classes')
parser.add_argument('--category_names', type=str, help='JSON file mapping labels to category names')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()
model, optimizer, epochs, categories, total_inputs, total_outputs = load_checkpoint(args.checkpoint)
top_p, classes = predict(args.img, model, args.top_k, args.category_names, args.gpu)
print(top_p)
print(classes)

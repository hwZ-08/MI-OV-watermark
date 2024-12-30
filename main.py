import argparse
import yaml

from train.classification import Classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for running DNN watermarking experiments")

    parser.add_argument('--action', default='watermark', choices=['clean', 'watermark', 'attack', 'evaluate'],
                        help='watermark: train a watermarked model,\n'
                             'clean: train a clean model,\n'
                             'attack: attack a given model,\n'
                             'evaluate: evaluate a given model')

    parser.add_argument('--dataset', default='cifar10', help='Dataset used to train a model (default: cifar10)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--data_path', default='./data', help='Path to store all the relevant datasets (default: ./data)')

    parser.add_argument('--arch', default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50', 'VGG16'],
                        help='Model architecture (default: resnet34)\n')

    parser.add_argument('--epochs', type=int, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--gpu_ids', default='0,1,2,3', type=str, help='GPU indices')

    parser.add_argument('--log_dir', default='./logs', help='Logging directory')
    parser.add_argument('--runname', default='', help='Name for an experiment')
    parser.add_argument('--save-interval', type=int, default=80, help='Save model interval')

    # evaluate a model, or resume from a model
    parser.add_argument('--checkpoint_path', help='Checkpoint path (example: xxx/xxx/last.pt)')

    # watermark
    parser.add_argument('--bits', default=512, type=int, help='Embedded bits')
    parser.add_argument('--wm_type', default='MemEnc', type=str, help='DNN Watermarking method')

    # watermark a pretrained model or train-from-sketch
    parser.add_argument('--clean_model_path', type=str, default=None, help='Path to the non-watermarked model')
    parser.add_argument('--from_pretrained', type=bool, default=True, help='Pretrain or train-from-sketch')
    parser.add_argument('--save_trigger', type=bool, default=True, help='Save the triggers')

    # attack a model
    parser.add_argument('--victim_path', default=None, help='Victim model path (example: xxx/xxx/last.pt)')
    parser.add_argument('--attack_type', default=None, choices=['misft', 'fine_prune', 'overwrite',
                                                                'ftal', 'rtal', 'prune'],
                        type=str, help='Attack type')

    parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as config_file:
            config_args = yaml.safe_load(config_file)
            for key, value in config_args.items():
                setattr(args, key, value)

    return args


if __name__ == '__main__':
    args = parse_args()

    classifier = Classifier(args)
    if args.action in ['clean', 'watermark']:
        classifier.train()
    elif args.action == 'attack':
        classifier.attack()
    else:
        classifier.evaluate()


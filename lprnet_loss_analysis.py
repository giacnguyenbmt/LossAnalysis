import argparse
import os

import cv2
import torch
import numpy as np
from torch.autograd import Variable

import utils
from source.loss_calculation import LossCalculator
from model import LPRNetEnhanceV20


img_w, img_h = (188, 24)

def preprocess(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_w, img_h))
        image = image.astype('float32')
        image -= 127.5
        image *= 0.0078125
        image = np.transpose(image, (2, 0, 1))
        img_batch = np.expand_dims(image, axis=0)
        tensor_data = torch.from_numpy(img_batch)
        return tensor_data


class LossAnalysis:
    def __init__(self, 
                 model, 
                 chars, 
                 t_length, 
                 blank_id=0) -> None:
        self.model = model
        self.chars=chars
        self.t_length=t_length
        blank_id=blank_id
        self.loss_calculator = LossCalculator(
            chars=chars,
            t_length=t_length,
            blank_id=blank_id
        )
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def get_loss_value_from_pp_dataset(self, 
                                       data_dir, 
                                       gt_path, 
                                       output_path):
        with open(gt_path) as f:
            content = f.readlines()

        f = open(output_path, 'w')
        f.write(
            '{},{}\n'.format(
                'image_file',
                'loss_value'
            )
        )

        for line in content:
            image_file, gt = line.strip().split()
            image_path = os.path.join(data_dir, image_file)
            img = cv2.imread(image_path)
            img = preprocess(img)
            images = Variable(img, requires_grad=False).to(self.device)
            logits = self.model(images)
            logits = logits.cpu().detach().numpy()

            gt_list = [gt]
            dt = logits # => N, C, T
            loss = self.loss_calculator.fit(gt_list, dt)
            text_to_file = '{},{}\n'.format(
                image_file,
                loss
            )
            f.write(text_to_file)
        f.close()


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to cal loss')
    parser.add_argument("-c", "--config", required=True, help="path to the config file",
                        type=str, default=None)
    parser.add_argument("-p", "--pretrained_model", help="path to the pretrained model",
                        type=str, default=None)
    parser.add_argument("--data_dir", help="path to the dataset",
                        type=str, default=None)
    parser.add_argument("--output_path", help="path to output_file",
                        type=str, default=None)
    parser.add_argument("-o", "--opt", nargs='+', help="set configuration options")
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()

    data_dir = args.data_dir
    output_path = args.output_path

    gt_path = os.path.join(data_dir, 'rec.txt')
    config = utils.parse_config(args)
    arch_config = config['Architecture']
    module_name = arch_config.pop('name')

    lpr = eval(module_name)(**arch_config)
    device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    lpr = utils.load_pretrained_model(lpr, config['Global']['pretrained_model'], device)
    print("Load pretrained model successful!")
    lpr.eval()
    torch.set_grad_enabled(False)

    analyzer = LossAnalysis(
        lpr, 
        chars=config['Global']['chars'], 
        t_length=18, 
        blank_id=config['Global']['blank_idx']
    )
    analyzer.get_loss_value_from_pp_dataset(data_dir, gt_path, output_path)

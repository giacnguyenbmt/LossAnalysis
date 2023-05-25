import torch

from .metrics.tf_ctc_loss import tf_ctcloss


class LossCalculator:
    def __init__(self, chars, t_length, blank_id=0) -> None:
        self.chars = list(chars)
        self.blank_id = blank_id
        self.t_length = t_length

        self.chars.insert(self.blank_id, '-')
        self.chars = ''.join(self.chars)
        self.char_dict = {char:i for i, char in enumerate(self.chars)}


    def _encode_label(self, label):
        encode = [self.char_dict[c] for c in label]
        return encode


    def _sparse_tuple_for_ctc(self, gt_list):
        labels = []
        input_lengths = []
        target_lengths = []
        for gt in gt_list:
            label = self._encode_label(gt)
            labels.extend(label)
            input_lengths.append(self.t_length)
            target_lengths.append(len(label))
        labels = torch.tensor(labels)
        input_lengths = tuple(input_lengths)
        target_lengths = tuple(target_lengths)
        return labels, input_lengths, target_lengths


    def fit(self, gt, dt):
        labels, input_lengths, target_lengths = self._sparse_tuple_for_ctc(gt)

        logits = torch.from_numpy(dt)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        # log_probs = log_probs.log_softmax(2)
        log_probs = log_probs.log()
        loss = tf_ctcloss(log_probs, 
                          labels, 
                          input_lengths, 
                          target_lengths, 
                          blank=self.blank_id)
        return loss.item()

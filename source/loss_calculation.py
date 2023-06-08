import numpy as np
import torch
import torch.nn.functional as F


def tf_ctcloss(log_probs,
               targets,
               input_lengths,
               target_lengths,
               blank=0,
               reduction='sum',
               zero_infinity=False):
    loss = F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity
    )
    if len(log_probs.shape) == 3:
        loss = loss / log_probs.shape[1]
    else:
        loss = loss / 1
    return loss


class LossCalculator:
    def __init__(self, chars, t_length, blank_id=0) -> None:
        self.chars = list(chars)
        self.blank_id = blank_id
        self.t_length = t_length

        self.chars.insert(self.blank_id, '-')
        self.chars = ''.join(self.chars)
        self.char_dict = {char:i for i, char in enumerate(self.chars)}


    def encode_label(self, label):
        encode = [self.char_dict[c] for c in label]
        return encode


    def sparse_tuple_for_ctc(self, gt_list):
        labels = []
        input_lengths = []
        target_lengths = []
        for gt in gt_list:
            label = self.encode_label(gt)
            labels.extend(label)
            input_lengths.append(self.t_length)
            target_lengths.append(len(label))
        labels = torch.tensor(labels)
        input_lengths = tuple(input_lengths)
        target_lengths = tuple(target_lengths)
        return labels, input_lengths, target_lengths


    def fit(self, gt, dt):
        labels, input_lengths, target_lengths = self.sparse_tuple_for_ctc(gt)

        logits = torch.from_numpy(dt)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        log_probs = log_probs.log_softmax(2)
        # log_probs = log_probs.log()
        loss = tf_ctcloss(log_probs, 
                          labels, 
                          input_lengths, 
                          target_lengths, 
                          blank=self.blank_id)
        return loss.item()


if __name__ == '__main__':
    chars = '0123456789ABCDĐEFGHKLMNPQRSTUVXYZ'
    foo = LossCalculator(chars=chars, t_length=18, blank_id=len(chars))

    # test_1
    # gt = ['11B128509']
    # dt = np.fromfile('input/output_sample_6e_5.raw', dtype=np.float32).reshape((1, 34, 18))
    # loss = foo.fit(gt, dt)
    # print('{:.5f}'.format(loss))

    # test_2
    # gt = ['11B128509', '11B128509', '11B128509']
    # dt = np.fromfile('input/output_sample_32e_4.raw', dtype=np.float32).reshape((3, 34, 18))
    # loss = foo.fit(gt, dt)
    # print('{:.5f}'.format(loss))

    # test_2
    gt = ['80A01566']
    dt = np.fromfile('input/output_sample_55e_3.raw', dtype=np.float32).reshape((1, 34, 18))
    loss = foo.fit(gt, dt)
    print('{:.5f}'.format(loss))

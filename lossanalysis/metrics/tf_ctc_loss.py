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

"""
        from :https://github.com/LiyuanLucasLiu/LightNER
    .. module:: crf
        :synopsis: conditional random field

    .. moduleauthor:: Liyuan Liu
"""

import torch
import torch.nn as nn
import numpy as np

class CRF_L(nn.Module):
    """
    Conditional Random Field Module
    Parameters
    ----------
    hidden_dim : ``int``, required.
        the dimension of the input features.
    tagset_size : ``int``, required.
        the size of the target labels.
    if_bias: ``bool``, optional, (default=True).
        whether the linear transformation has the bias term.
    """
    def __init__(self, 
                hidden_dim: int, 
                tagset_size: int, 
                if_bias: bool = True):
        super(CRF_L, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))

    def rand_init(self):
        """
        random initialization
        """
        init_linear(self.hidden2tag)
        self.transitions.data.zero_()

    def forward(self, feats):
        """
        calculate the potential score for the conditional random field.
        Parameters
        ----------
        feats: ``torch.FloatTensor``, required.
            the input features for the conditional random field, of shape (*, hidden_dim).
        Returns
        -------
        output: ``torch.FloatTensor``.
            A float tensor of shape (ins_num, from_tag_size, to_tag_size)
        """

        tmp = feats.size()
        batch_size = tmp[0]
        seq_len = tmp[1]
        scores = self.hidden2tag(feats).view(-1, 1, self.tagset_size)
        ins_num = scores.size(0)
        crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size) + self.transitions.view(1, self.tagset_size, self.tagset_size).expand(ins_num, self.tagset_size, self.tagset_size)

        return crf_scores.view(-1,seq_len,self.tagset_size,self.tagset_size)


class CRFLoss_vb(nn.Module):
    """loss for viterbi decode

    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch

    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=False):
        super(CRFLoss_vb, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

    def forward(self, scores, target, mask):
        """
        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : crf scores
            target (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """

        if torch.cuda.is_available():
            scores = scores.transpose(1, 0).cuda()
            target = target.transpose(1, 0).unsqueeze(2).cuda()
            mask = mask.byte().transpose(1, 0).cuda()
        else:
            scores = scores.transpose(1, 0)
            target = target.transpose(1, 0).contiguous().unsqueeze(2)
            mask= mask.byte().transpose(1, 0).contiguous()

        # calculate batch size and seq len
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        # calculate sentence score
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target).view(seq_len, bat_size)  # seq_len * bat_size
        tg_energy = tg_energy.masked_select(mask).sum()

        # calculate forward partition score

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_tag, :].clone()  # bat_size * to_target_size
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = log_sum_exp(cur_values, self.tagset_size)
            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(bat_size, 1).expand(bat_size, self.tagset_size)
            partition.masked_scatter_(mask_idx, cur_partition.masked_select(mask_idx))  #0 for partition, 1 for cur_partition

        #only need end at end_tag
        partition = partition[:, self.end_tag].sum()
        # average = mask.sum()
        # average_batch
        if self.average_batch:
            loss = (partition - tg_energy) / bat_size
        else:
            loss = (partition - tg_energy)

        return loss

class CRFDecode_vb():
    """Batch-mode viterbi decode

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch

    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

    def decode(self, scores, mask):
        """Find the optimal path with viterbe decode

        args:
            scores (size seq_len, bat_size, target_size_from, target_size_to) : crf scores
            mask (seq_len, bat_size) : mask for padding
        return:
            decoded sequence (size seq_len, bat_size)
        """
        # calculate batch size and seq len

        if torch.cuda.is_available():
            scores = scores.transpose(1, 0).cuda()
            mask = mask.byte().transpose(1, 0).cuda()
        else:
            scores = scores.transpose(1, 0)
            mask= mask.byte().transpose(1, 0).contiguous()

        seq_len = scores.size(0)
        bat_size = scores.size(1)

        mask = 1 - mask
        decode_idx = torch.LongTensor(seq_len-1, bat_size)

        # calculate forward score and checkpoint

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        forscores = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        back_points = list()
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)

            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)
            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points)-2, -1, -1):
            back_point = back_points[idx]
            index = pointer.contiguous().view(-1,1)
            pointer = torch.gather(back_point, 1, index).view(-1)
            decode_idx[idx] = pointer
        return decode_idx

def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M


class CRFLoss_vb_PA(nn.Module):
    """loss for viterbi decode
    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )
    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch

    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=False):
        super(CRFLoss_vb_PA, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch
        self.NINF=-100000

    def forward(self, scores, target, mask):
        """
        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : crf scores
            target (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """

        if torch.cuda.is_available():
            scores = scores.transpose(1, 0).cuda()
            target = target.transpose(1, 0).cuda()
            mask = mask.byte().transpose(1, 0).cuda()
        else:
            scores = scores.transpose(1, 0)
            target = target.transpose(1, 0).contiguous()
            mask= mask.byte().transpose(1, 0).contiguous()


        # calculate batch size and seq len
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_tag, :].clone()  # bat_size * to_target_size
        tag_partition = inivalues[:, self.start_tag, :].clone().masked_fill_(target[0],self.NINF)

        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target

            cur_tag_values = cur_values + tag_partition.contiguous().view(bat_size, self.tagset_size, 1).expand(
                bat_size, self.tagset_size, self.tagset_size)
            cur_tag_partition = log_sum_exp(cur_tag_values, self.tagset_size).masked_fill_(target[idx],self.NINF)
            cur_values = cur_values + partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size,
                                                                                                        self.tagset_size,
                                                                                                        self.tagset_size)
            cur_partition = log_sum_exp(cur_values, self.tagset_size)
            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            mask_idx = mask[idx, :].view(bat_size, 1).expand(bat_size, self.tagset_size)
            tag_partition.masked_scatter_(mask_idx, cur_tag_partition.masked_select(mask_idx))
            partition.masked_scatter_(mask_idx, cur_partition.masked_select(mask_idx))
            # partition = new_partition

        # only need end at end_tag
        tag_partition = tag_partition[:, self.end_tag]
        tp_mask = (tag_partition == self.NINF)
        # target = tag_partition.masked_select(tp_mask).sum()
        target = tag_partition.masked_fill_(tp_mask, 0).sum()
        partition = partition[:, self.end_tag].sum()
        # average = mask.sum()

        # average_batch
        if self.average_batch:
            loss = (partition - target) / bat_size
        else:
            loss = (partition - target)

        return loss


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()
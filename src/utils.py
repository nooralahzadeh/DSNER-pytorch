
import os
import torch
from vocab import Vocab
import math
from tqdm import tqdm

from typing import List, Tuple
import re


def load_word_vectors(path):
    if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
        vectors = torch.load(path + '.pth')
        vocab = Vocab(filename=path + '.vocab')
        return vocab, vectors

    print("==> Preprocessing glove file!")
    count = sum(1 for _ in open(path + '.txt'))
    with open(path + '.txt', 'r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None] * count
    vectors = torch.zeros(count, dim)
    with open(path + '.txt', 'r') as f:
        idx = 0
        for line in tqdm(f):
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            vectors[idx] = torch.Tensor(list(map(float, contents[1:])))
            idx += 1
    with open(path + '.vocab', 'w') as f:
        for word in words:
            f.write(word + '\n')
    vocab = Vocab(filename=path + '.vocab')
    torch.save(vectors, path + '.pth')
    return vocab, vectors

def load_word_vectors_EC(path):
    if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
        vectors = torch.load(path + '.pth')
        vocab = Vocab(filename=path + '.vocab')
        return vocab, vectors

    print("==> Preprocessing glove file!")
    count = sum(1 for _ in open(path + '.txt'))
    with open(path + '.txt', 'r') as f:
        contents = f.readline().rstrip('\n').split('\t')
        dim = len(contents[1:])
    words = [None] * count
    vectors = torch.zeros(count, dim)
    with open(path + '.txt', 'r') as f:
        idx = 0
        for line in tqdm(f):
            contents = line.rstrip('\n').split('\t')
            words[idx] = contents[0]
            vectors[idx] = torch.Tensor(list(map(float, contents[1:])))
            idx += 1
    with open(path + '.vocab', 'w') as f:
        for word in words:
            f.write(word + '\n')
    vocab = Vocab(filename=path + '.vocab')
    torch.save(vectors, path + '.pth')
    return vocab, vectors


def build_vocab(filenames, vocabfile, char=False,lowercase=False):
    vocab = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                if char:
                    words=line.rstrip('\n').split()
                    tokens=[]
                    for word in words:
                        for ch in word:
                            tokens.append(ch)
                else:
                    tokens = line.rstrip('\n').split()
                vocab |= set(tokens)
    with open(vocabfile, 'w') as f:
        for token in sorted(vocab):
            f.write(token + '\n')

def write_output(filename,result):
    with open(filename, 'w') as f:
        for key, value in result.items():
            f.write('%s:%s\n' % (key, value))





def map_label_to_target(label):
    target = torch.LongTensor(1)
    target[0] = int(label)
    return target


def map_float_label_to_target(label, num_classes=5):
    target = torch.zeros(1, num_classes)
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0][floor - 1] = 1
    else:
        target[0][floor - 1] = ceil - label
        target[0][ceil - 1] = label - floor
    return target


def count_params(model):
    print("__param count_")
    params = list(model.parameters())
    total_params = 0
    for p in params:
        if p.requires_grad:
            total_params += p.numel()
            print(p.size())
    print("total", total_params)
    print('______________')

def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def revlut(lut):
    return {v: k for k, v in lut.items()}

def log_sum_exp_PA(x,islast=False):
    pass

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

# Turn a sequence of IOB chunks into single tokens
def iob_to_spans(sequence, lut, strict_iob2=False):
    """
    convert to iob to span
    """
    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = lut[y]

        if label.startswith('B-'):
            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i]

        elif label.startswith('I-'):

            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning, type=IOB2, unexpected format ([%s] follows other tag type [%s] @ %d)' % (
                            label, current[0], i))

                    current = [base, '%d' % i]

            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning, unexpected format (I before B @ %d) %s' % (i, label))
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)

# Turn a sequence of IOBES chunks into single tokens
def iobes_to_spans(sequence, lut, strict_iob2=False):
    """
    convert to iobes to span
    """
    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = lut[y]

        if label.startswith('B-'):

            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i]

        elif label.startswith('S-'):

            if current is not None:
                chunks.append('@'.join(current))
                current = None
            base = label.replace('S-', '')
            chunks.append('@'.join([base, '%d' % i]))

        elif label.startswith('I-'):

            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning')
                    current = [base, '%d' % i]

            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning')

        elif label.startswith('E-'):

            if current is not None:
                base = label.replace('E-', '')
                if base == current[0]:
                    current.append('%d' % i)
                    chunks.append('@'.join(current))
                    current = None
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning')
                    current = [base, '%d' % i]
                    chunks.append('@'.join(current))
                    current = None

            else:
                current = [label.replace('E-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning')
                chunks.append('@'.join(current))
                current = None
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)


def sequence_mask(lens: torch.Tensor, max_len: int = None) -> torch.ByteTensor:
    """
    Compute sequence mask.

    Parameters
    ----------
    lens : torch.Tensor
        Tensor of sequence lengths ``[batch_size]``.

    max_len : int, optional (default: None)
        The maximum length (optional).

    Returns
    -------
    torch.ByteTensor
        Returns a tensor of 1's and 0's of size ``[batch_size x max_len]``.

    """
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().item()

    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = torch.autograd.Variable(ranges)

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask


def pad(tensor: torch.Tensor, length: int) -> torch.Tensor:
    """Pad a tensor with zeros."""
    return torch.cat([
        tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()
    ])


def sort_and_pad(tensors: List[torch.Tensor],
                 lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort and pad list of tensors by their lengths and concatenate them.

    Parameters
    ----------
    tensors : List[torch.Tensor]
        The list of tensors to pad, each has dimension ``[L x *]``, where
        ``L`` is the variable length of each tensor. The remaining dimensions
        of the tensors must be the same.

    lengths : torch.Tensor
        A tensor that holds the length of each tensor in the list.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The first tensor is the result of concatenating the list and has
        dimension ``[len(tensors) x max_length x *]``.
        The second tensor contains the sorted lengths from the original list.
        The third tensor contains the sorted indices from the original list.

    """
    sorted_lens, sorted_idx = lengths.sort(0, descending=True)
    max_len = sorted_lens[0].item()
    padded = []
    for i in sorted_idx:
        padded.append(pad(tensors[i], max_len).unsqueeze(0))
    return torch.cat(padded), sorted_lens, sorted_idx


def unsort(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Unsort a tensor along dimension 0."""
    unsorted = tensor.new_empty(tensor.size())
    unsorted.scatter_(0, indices.unsqueeze(-1).expand_as(tensor), tensor)
    return unsorted


def assert_equal(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> None:
    """Check if two tensors are equal."""
    assert tensor_a.size() == tensor_b.size()
    assert (tensor_a == tensor_b).all().item() == 1



#####
def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O' or tag == 'UNK':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


###
def normalizeParentheses(line):
    words=line.split()
    new=[]
    for str_word in words:
        if str_word == '(':
            str_word = '-LRB-'
            new.append(str_word)
        if str_word == ')':
            str_word = '-RRB-'
            new.append(str_word)
        else:
            new.append(str_word)
    return " ".join(new)

def reverse_normalizeParentheses(tokens):
    new=[]
    for str_word in tokens:
        if str_word == '-LRB-':
            str_word = '('
            new.append(str_word)
        if str_word == '-RRB-':
            str_word = ')'
            new.append(str_word)
        else:
            new.append(str_word)
    return new


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
import unicodedata

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

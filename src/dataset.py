

import os, random
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

import constants as C

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EC_PA_Datset(data.Dataset):
    def __init__(self, path, word_vocab, tags_vocab, tags_iobes_vocab, partial=False, treetype="", cased=""):
        super(EC_PA_Datset, self).__init__()
        self.maxlen = 75
        self.word_vocab = word_vocab
        self.partial=partial
        self.sign= 1 if 'pa' in path else 0
        self.tags_vocab = tags_vocab
        self.tags_vocab_iobes= tags_iobes_vocab
        self.num_tags = {'iob':tags_vocab.size(),'iobes':tags_iobes_vocab.size()}
        self.sentences, self.signs, self.lengths = self.read_sentences(os.path.join(path, 'a%s%s.txt' % (treetype,cased)))
        self.tags, self.tags_one_hot = self.read_tags(os.path.join(path, 'tags%s.txt' % treetype))
        self.size = len(self.tags)
        self.tags_iobes, self.tags_iobes_one_hot = self.read_tags(os.path.join(path, 'tags-iobes%s.txt' % treetype),
                                                                  iobes=True)

    def __len__(self):
        return self.size



    def split(self,indices):
        self.sentences=self.sentences[indices]
        self.signs=self.signs[indices]
        self.lengths=self.lengths[indices]
        self.tags=self.tags[indices]
        self.tags_iobes=self.tags_iobes[indices]
        self.tags_one_hot = self.tags_one_hot[indices]
        self.tags_iobes_one_hot = self.tags_iobes_one_hot[indices]
        self.size=len(self.tags)
        return self




    def add_item (self, sent, tags,tags_one_hot,tags_iobes_one_hots, length):
        self.sentences=torch.cat([self.sentences,sent.unsqueeze(0)], dim=0)
        self.tags=torch.cat([self.tags,tags.unsqueeze(0)], dim=0)
        self.tags_one_hot = torch.cat([self.tags_one_hot, tags_one_hot.unsqueeze(0)], dim=0)
        self.tags_iobes_one_hots = torch.cat([self.tags_iobes_one_hots, tags_iobes_one_hots.unsqueeze(0)], dim=0)
        self.lengths =torch.cat([self.lengths,length], dim=0)
        self.size=self.size+1

        return self

    def __getitem__(self, index):
        sent = deepcopy(self.sentences[index])
        tags=deepcopy(self.tags[index])
        tags_one_hot=deepcopy(self.tags_one_hot[index])
        tags_iobes=deepcopy(self.tags_iobes[index])
        tags_iobes_one_hots=deepcopy(self.tags_iobes_one_hot[index])

        #lm_input=deepcopy(self.sentences_words_in_chars[index])
        sign=deepcopy(self.signs[index])
        s_length=deepcopy(self.lengths[index])

        return  sent, tags, tags_iobes, sign, s_length,tags_one_hot,tags_iobes_one_hots


    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences= [self.read_sentence(line) for line in tqdm(f.readlines())]

        signs=[self.sign for i in sentences]

        sentences = list(
            map(lambda s: list(map(lambda w: self.word_vocab.label_to_idx.get(w, self.word_vocab.label_to_idx[C.UNK_WORD]), s)), sentences))

        #self.maxlen = max(list(map(lambda s: len(s), sentences)))
        # pad
        sentences_padded = []
        lengths=[]
        for sentence in sentences:
            lengths.append(len(sentence)+1)
            sentences_padded.append(sentence + [self.word_vocab.label_to_idx[C.PAD_WORD]] * (self.maxlen - len(sentence)))
        return torch.LongTensor(sentences_padded), torch.LongTensor(signs), torch.LongTensor(lengths),

    def read_sentence(self, line):
        words=[w for w in line.rstrip('\n').split('\t')]
        return words

    def read_tags(self, filename, iobes=False):
        if iobes:
            tags_vocab=self.tags_vocab_iobes.label_to_idx
        else:
            tags_vocab = self.tags_vocab.label_to_idx
        with open(filename, 'r') as f:
            tags = [self.read_tag(line) for line in tqdm(f.readlines())]

        # Encode tags into tag maps with <end> at the end
        tmaps = list(map(lambda s: [tags_vocab[C.BOS_WORD]]+list(map(lambda t: tags_vocab[t], s)) , tags))

        # Since we're using CRF scores of size (prev_tags, cur_tags), find indices of target sequence in the unrolled scores
        # This will be row_index (i.e. prev_tag) * n_columns (i.e. tagset_size) + column_index (i.e. cur_tag)

        labels_for_encoding=[]
        UNK = tags_vocab['O']
        possible_tags=[tags_vocab[t] for t in tags_vocab if t not in [C.PAD_WORD,C.BOS_WORD]]

        for tmap in tmaps:
            label=[]
            for t  in tmap:
                if self.sign == 1 and self.partial and  t== UNK:
                    label.append(possible_tags)
                else:
                    label.append([t])
            labels_for_encoding.append(label)

        tag_pad_len = self.maxlen
        tmaps_one_hot = []
        for label_for_encoding in labels_for_encoding:
            one_hot=self.label_encoding(label_for_encoding,len(tags_vocab),tags_vocab[C.PAD_WORD],self.maxlen)
            tmaps_one_hot.append(one_hot)

        tmaps = list(
            map(lambda s: [s[i] * len(tags_vocab) + s[i+1] for i in range(0, len(s)-1)]+[s[len(s)-1]* len(tags_vocab) +tags_vocab[C.PAD_WORD]], tmaps))
        #+[s[len(s)]* len(tags_vocab) +tags_vocab[C.PAD_WORD]]
        # Sanity check

        padded_tmaps=[]
        for tmap in tmaps:
            padded_tmaps.append(tmap + [tags_vocab[C.PAD_WORD]] * (tag_pad_len - len(tmap)))

        return torch.LongTensor(padded_tmaps),torch.LongTensor(tmaps_one_hot)

    def read_tag(self, line):
        indices=[t for t in line.rstrip('\n').split('\t')]
        return indices

    def label_encoding(self,label, label_size, pad_label, tot_len):
        """
        pad label for vtb decode format
        """
        encoded_label = [[1] * label_size for ind in range(tot_len)]
        cur_len = len(label) - 1
        for ind in range(0, cur_len):
            for ind_label in label[ind + 1]:
                encoded_label[ind][ind_label] = 0
        for ind in range(cur_len, tot_len):
            encoded_label[ind][pad_label] = 0
        return encoded_label

    def merge(self, dataset):

        sentences_padded = []
        for sentence in dataset.sentences.tolist():
            sentences_padded.append(
                sentence + [self.word_vocab.label_to_idx[C.PAD_WORD]] * (self.maxlen - len(sentence)))

        sentences=torch.LongTensor(sentences_padded)
        self.sentences=torch.cat([self.sentences,sentences], dim=0)

        self.signs=torch.cat([self.signs,dataset.signs], dim=0)

        padded_tmaps=[]

        for tmap in dataset.tags.tolist():
            padded_tmaps.append(tmap + [self.tags_vocab.label_to_idx[C.PAD_WORD]] * (self.maxlen - len(tmap)))

        tags = torch.LongTensor(padded_tmaps)
        self.tags=torch.cat([self.tags, tags], dim=0)

        padded_tmaps_one_hot = []

        for tmap in dataset.tags_one_hot.tolist():
            padded_tmaps_one_hot.append(tmap + [self.tags_vocab.label_to_idx[C.PAD_WORD]] * (self.maxlen - len(tmap)))

        tags = torch.LongTensor(padded_tmaps_one_hot)
        self.tags_one_hot = torch.cat([self.tags_one_hot, tags], dim=0)
        self.size += dataset.size

        padded_tmaps = []

        for tmap in dataset.tags_iobes.tolist():
            padded_tmaps.append(tmap + [self.tags_vocab_iobes.label_to_idx[C.PAD_WORD]] * (self.maxlen - len(tmap)))

        tags = torch.LongTensor(padded_tmaps)
        self.tags_iobes=torch.cat([self.tags_iobes,tags], dim=0)

        padded_tmaps = []
        for tmap in dataset.tags_iobes_one_hot.tolist():
            padded_tmaps.append(tmap + [self.tags_vocab_iobes.label_to_idx[C.PAD_WORD]] * (self.maxlen - len(tmap)))

        tags = torch.LongTensor(padded_tmaps)
        self.tags_iobes_one_hot = torch.cat([self.tags_iobes_one_hot, tags], dim=0)


        self.lengths=torch.cat([self.lengths,dataset.lengths], dim=0)
        return self

    def pad(self,input):
        for i in range(0, self.maxlen - len(input)):
            input.append(C.PAD_WORD)
        return input


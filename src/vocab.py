"""
Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
A copy of the License is located at

http://www.apache.org/licenses/LICENSE-2.0

or in the "license" file accompanying this file. This file is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License.
"""
import string
class Vocab(object):
    def __init__(self, filename=None, data=None, lower=False):
        self.idx_to_label = {}
        self.label_to_idx = {}
        self.lower = lower
        self.special = []

        if data is not None:
            self.add_specials(data)

        if filename is not None:
            self.load_file(filename)

        else:
            self.char_vocab()


    def size(self):
        return len(self.idx_to_label)

    def add_specials(self, labels):
        """
        Handle the special labels such as <unk>, <pad>

        :param labels: list of labels
        """
        for label in labels:
            idx = self.add(label)
            self.special += [idx]

    def load_file(self, filename):
        """
        :param filename: vocab file
        """
        with open(filename, 'r') as infile:
            for line in infile:
                token = line.rstrip('\n')
                self.add(token)

    def char_vocab(self):
        for char in string.ascii_letters + string.digits + string.punctuation:
            self.add(char)
            self.add(char.upper())

    def vocab_from_dictinay(self,list_of_items):
        for l in list_of_items:
            self.add(l)

    def add(self, label):
        """
        Add the label into vocabulary

        :param label:
        :return: index
        """

        label = label.lower() if self.lower else label

        if label in self.label_to_idx:
            idx = self.label_to_idx[label]
        else:
            idx = len(self.idx_to_label)
            self.idx_to_label[idx] = label
            self.label_to_idx[label] = idx
        return idx

    def get_index(self, label, default=None):
        """
        :param label:
        :param default:
        :return: index
        """
        label = label.lower() if self.lower else label
        if label in self.label_to_idx:
            return self.label_to_idx[label]
        else:
            return default

    def convert_to_idx(self, labels, unk_word):
        """
        Convert labels into indices.
        If a label is not in vocabulary, return unknown word index

        :param labels: list of labels
        :param unk_word: unknown word
        :return: list of indices
        """
        vec = []
        unk = self.get_index(unk_word)
        vec += [self.get_index(label, default=unk) for label in labels]
        return vec

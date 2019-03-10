

from utils import sequence_mask
from crf_LightNER import *
torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import constants as C
class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))



class policy_selector(nn.Module):
    def __init__(self,input_size,
                 hidden_size):
        super(policy_selector, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        action_scores = self.affine2(x)
        return self.sigmoid(action_scores)
        return y_pred



class BiLSTM_CRF_PA_CRF_LightNER(nn.Module):
    def __init__(self, vocab,
                 emb_size,
                 tags_vocab,
                 freeze_embed,
                 biLSTM_hidden_size=None,
                 dropout=None):
        """ """
        super(BiLSTM_CRF_PA_CRF_LightNER, self).__init__()
        # word embedding

        self.vocab_size=len(vocab)
        padding_idx_in_sentence = vocab['<PAD>']

        self.embeddings = nn.Embedding(self.vocab_size, emb_size, padding_idx=padding_idx_in_sentence)

        if freeze_embed:  # no update
            self.embeddings.weight.requires_grad = False

        self.dropout = nn.Dropout(p=dropout) if dropout!= -1 else None


        self.tagset_size = len(tags_vocab)
        self.tags_vocab=tags_vocab


        # Hierachical Bi-LSTM
        self.biLSTM_hiddes_size=biLSTM_hidden_size

        self.BiLSTM = nn.LSTM(
            input_size=emb_size,
            hidden_size=self.biLSTM_hiddes_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        self.crf_layer = CRF_L(2* self.biLSTM_hiddes_size, self.tagset_size)
        self.crf_layer.rand_init()
        self.viterbiLoss = CRFLoss_vb( self.tagset_size,self.tags_vocab[C.BOS_WORD],self.tags_vocab[C.PAD_WORD])
        self.viterbiLoss_PA= CRFLoss_vb_PA(self.tagset_size, self.tags_vocab[C.BOS_WORD], self.tags_vocab[C.PAD_WORD])
        self.viterbiDecoder = CRFDecode_vb(self.tagset_size,self.tags_vocab[C.BOS_WORD],self.tags_vocab[C.PAD_WORD])

    def _feature(self,inputs,lengths,tags_one_hot=None):
        """"""
        w_lengths, word_sort_ind = lengths.sort(dim=0, descending=True)
        # should catch from  proper index
        inputs = inputs[word_sort_ind].to(device)
        if tags_one_hot is not None:
            tags_one_hot = tags_one_hot[word_sort_ind].byte().to(device)

        # compute features
        inputs_emb = self.embeddings(inputs)
        w = self.dropout(inputs_emb)

        # Pack padded sequence
        w = torch.nn.utils.rnn.pack_padded_sequence(w, list(w_lengths),
                                                    batch_first=True)  # packed sequence of word_emb_dim + 2 * char_rnn_dim, with real sequence lengths

        # LSTM
        w, _ = self.BiLSTM(w)  # packed sequence of word_rnn_dim, with real sequence lengths
        # Unpack packed sequence

        w, _ = torch.nn.utils.rnn.pad_packed_sequence(w,
                                                      batch_first=True)  # (batch_size, max_word_len_in_batch, word_rnn_dim)

        w = self.dropout(w)

        mask = sequence_mask(w_lengths).float()

        crf_scores = self.crf_layer(w)
        return  crf_scores, tags_one_hot, mask, w_lengths, word_sort_ind

    def forward_eval(self,inputs,lengths,tags):
        crf_scores, _, mask,w_lengths, word_sort_ind  = self._feature(inputs, lengths)
        loss = self.viterbiLoss(crf_scores, tags, mask)
        tmaps = tags[word_sort_ind] % self.tagset_size  # actual target indices (see dataset())

        preds = self.viterbiDecoder.decode(crf_scores, mask)
        return loss, preds, w_lengths, tmaps, word_sort_ind

    def forward(self,inputs, lengths, tags_one_hot):
        crf_scores, tags_one_hot, mask,_,_ =self._feature(inputs,lengths,tags_one_hot)
        loss = self.viterbiLoss_PA(crf_scores, tags_one_hot, mask)
        return  loss





from tqdm import tqdm
import torch
import math
import numpy as np
import torch.nn.functional as F
import constants as C
from conlleval import tags_to_labels,evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Trainer_PA_SL_DSNER(object):
    """"""
    def __init__(self, args, tagger_mdl, sl_mdl, optimizer_tagger, optimizer_sl, criterion_sl, partial):
        """"""
        self.args = args
        self.tagger_model = tagger_mdl
        self.sl_model = sl_mdl
        self.criterion_sl = criterion_sl

        self.tagset_size = tagger_mdl.tagset_size
        self.o_tag = tagger_mdl.tags_vocab['O']
        self.pad_tag = tagger_mdl.tags_vocab[C.PAD_WORD]

        '''
        it need to define two optimizer:
            1- for selector
            2- for encoder+crf

        first, calculate the reward from encoder+crf
        then,  calculate loss for the selector
        and,  add reward to the loss and optimize the select parameters 
        for name, param in self.model.named_parameters():
                    print (f'{name}: {param.requires_grad}')
        '''
        #

        self.optimizer_tagger = optimizer_tagger
        self.optimizer_sl = optimizer_sl
        self.epoch = 0
        self.partial = partial

    def train(self, dataset, epoch):
        """"""
        self.tagger_model.train()
        # self.optimizer_tagger.zero_grad()

        self.epoch = epoch
        total_loss = 0.0

        cnt = 0
        exist_num_in_batch = 0  # instance num in this batch(expert + select_pa_sample)
        PA_num_in_batch = 0  # select_pa_sample_num in this sample (number of PAs)
        y_label_in_batch = []  # all_pa_data action (0/1)
        PA_sentense_representation = []  # representation of every PA instance
        X_char_batch = []
        y_batch = []
        # sign_batch=[]
        s_length_batch = []
        y_one_hot_batch = []

        # not update
        total_PA_num = 0
        X_char_all = []
        y_all = []
        # sign_all=[]
        s_length_all = []
        y_one_hot_all = []

        indices = torch.randperm(len(dataset))

        for start_index in tqdm(range(len(dataset)), desc='Training epoch  ' + str(self.epoch) + ''):

            sent, tags, tags_iobes, sign, s_length, y_one_hot, y_iobes_one_hot = dataset[indices[start_index]]

            if self.args.iobes:
                tags = tags_iobes
                y_one_hot = y_iobes_one_hot
                del y_iobes_one_hot, tags_iobes
            else:
                del y_iobes_one_hot, tags_iobes


            if exist_num_in_batch == self.args.batch_size:  # if the number of the one that selected + experts== batch_size
                X_char_all.extend(X_char_batch)
                y_all.extend(y_batch)
                s_length_all.extend(s_length_batch)
                # sign_all.extend(sign_batch)
                y_one_hot_all.extend(y_one_hot_batch)
                cnt += 1
                '''
                    optimize the selector:
                    1. count reward: add all p(y|x) of dev_dataset and average
                    2. input1: average_reward
                    3. input2: all PA_sample in this step(0 or 1)
                '''
                if len(y_label_in_batch) > 0:
                    # calculate reward as r=1/(|A_i| +|H_i|) *(sum(log p(z|x)) + sum(log p(y|x)) just for EXperts and PA that selector choose
                    reward = self.get_reward(X_char_batch, s_length_batch, y_one_hot_batch)
                    reward_list = [reward for i in range(len(y_label_in_batch))]
                    # @todo convert to cuda:
                    self.optimize_selector(PA_sentense_representation, y_label_in_batch, reward_list)

                total_PA_num += PA_num_in_batch
                exist_num_in_batch = 0
                PA_num_in_batch = 0
                y_label_in_batch = []
                PA_sentense_representation = []
                X_char_batch = []
                y_batch = []
                sign_batch = []
                s_length_batch = []
                y_one_hot_batch = []

            if sign == 0:  # if it is the expert instance
                exist_num_in_batch += 1
                X_char_batch.append(sent)
                y_batch.append(tags)
                # sign_batch.append(sign)
                s_length_batch.append(s_length)
                y_one_hot_batch.append(y_one_hot)

            elif sign == 1:  # the PA instance
                this_representation = self.get_representation(sent, tags, s_length)
                PA_sentense_representation.append(this_representation)
                action_point = self.select_action(
                    this_representation)  # Get the probablity fro selector for the sentence
                if action_point > 0.5:
                    X_char_batch.append(sent)
                    y_batch.append(tags)
                    # sign_batch.append(sign)
                    s_length_batch.append(s_length)
                    y_one_hot_batch.append(y_one_hot)
                    PA_num_in_batch += 1
                    exist_num_in_batch += 1
                    y_label_in_batch.append(1)
                else:
                    y_label_in_batch.append(0)

        if exist_num_in_batch <= self.args.batch_size and exist_num_in_batch > 0:
            cnt += 1
            left_size = self.args.batch_size - exist_num_in_batch
            for i in range(left_size):
                index = np.random.randint(len(dataset))
                sent, tags, tags_iobes, sign, s_length, y_one_hot, y_one_hot_iobes = dataset[index]
                if self.args.iobes:
                    tags = tags_iobes
                    y_one_hot=y_one_hot_iobes
                X_char_batch.append(sent)
                y_batch.append(tags)
                sign_batch.append(sign)
                s_length_batch.append(s_length)
                y_one_hot_batch.append(y_one_hot)
            X_char_all.extend(X_char_batch)
            y_all.extend(y_batch)
            # sign_all.extend(sign_batch)
            s_length_all.extend(s_length_batch)
            y_one_hot_all.extend(y_one_hot_batch)

            if len(y_label_in_batch) > 0:
                reward = self.get_reward(X_char_batch, s_length_batch, y_one_hot_batch)
                reward_list = [reward for i in range(len(y_label_in_batch))]
                # just for PA (0,1)
                self.optimize_selector(PA_sentense_representation, y_label_in_batch, reward_list)


        # optimize baseline
        num_iterations = int(math.ceil(1.0 * len(X_char_all) / self.args.batch_size))
        total_loss = 0

        for iteration in tqdm(range(num_iterations), desc='Tagger optimizing  epoch  ' + str(self.epoch) + ''):
            self.tagger_model.train()
            self.optimizer_tagger.zero_grad()
            X_char_train_batch, s_length_batch, y_one_hot_batch = nextBatch(X_char_all, s_length_all, y_one_hot_all,
                                                                                           start_index=iteration * self.args.batch_size,
                                                                                           batch_size=self.args.batch_size)

            X_char_train_batch = torch.stack(X_char_train_batch)
            #y_train_batch = torch.stack(y_train_batch)
            s_length_batch = torch.stack(s_length_batch)
            y_one_hot_batch = torch.stack(y_one_hot_batch)

            batch_loss = self.tagger_model(X_char_train_batch, s_length_batch, y_one_hot_batch)

            total_loss += float(batch_loss)
            # retain_graph=True
            batch_loss.backward()
            self.optimizer_tagger.step()

        total_loss = float(total_loss / num_iterations)
        cnt += 1
        print("epoch %d iteration %d end, train_PA loss: %.4f, total_PA_num: %5d" % (
        epoch, cnt, total_loss, total_PA_num))
        # Decay learning rate every epoch
        return total_loss

    def get_reward(self, x_words, lengths, y_tags_one_hot):
        self.tagger_model.eval()
        with torch.no_grad():
            x_words = torch.stack(x_words)
            lengths = torch.stack(lengths)
            #y_tags=torch.stack(y_tags)
            y_tags_one_hot = torch.stack(y_tags_one_hot)
            batch_loss = self.tagger_model(x_words, lengths, y_tags_one_hot)
            reward = -1 * (batch_loss / self.args.batch_size)
        return reward

    def select_action(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0)
        self.sl_model.eval()
        prob = self.sl_model(state)

        return prob.item()

    def optimize_selector(self, x_representations, y_select, rewards):
        self.sl_model.train()
        self.optimizer_sl.zero_grad()
        #eps = np.finfo(np.float32).eps.item()

        x_representations = torch.stack(x_representations).to(device)
        y_select = torch.FloatTensor(y_select).to(device)

        y_preds = self.sl_model(x_representations)
        y_preds = y_preds.squeeze(2).squeeze(1)

        neg_log_prob = self.criterion_sl(y_preds, y_select)
        rewards = torch.FloatTensor(rewards).to(device)
        policy_loss = torch.sum(neg_log_prob * rewards)
        policy_loss = neg_log_prob * rewards

        lambda1, lambda2 = 0.003, 0.003
        all_linear1_params = torch.cat([x.view(-1) for x in self.sl_model.affine1.parameters()])
        all_linear2_params = torch.cat([x.view(-1) for x in self.sl_model.affine1.parameters()])
        l1_regularization = lambda1 * torch.norm(all_linear1_params, 1)
        l2_regularization = lambda2 * torch.norm(all_linear2_params, 2)
        policy_loss += l1_regularization + l2_regularization
        torch.autograd.backward(policy_loss,grad_tensors=torch.ones_like(policy_loss))
        #policy_loss.backward()
        self.optimizer_sl.step()

    def get_representation(self, X_char, y_char, length):
        # concat sample_representation and tag_representation
        # Get representation from BiLSTM for the current sentence
        with torch.no_grad():
            X_char = X_char.to(device)
            w = self.tagger_model.embeddings(X_char).unsqueeze(0)
            w = self.tagger_model.dropout(w)

            # Pack padded sequence
            w = torch.nn.utils.rnn.pack_padded_sequence(w, list([self.args.max_len]),
                                                        batch_first=True)
            # lstm
            output, (h, cell) = self.tagger_model.BiLSTM(
                w)  # packed sequence of word_rnn_dim, with real sequence lengths

            # Unpack packed sequence
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                               batch_first=True)  # (batch_size, max_word_len_in_batch, word_rnn_dim)

            output = self.tagger_model.dropout(output)

            w_last_node = output[:, length - 1, :]

            this_representation_o_t = self.tagger_model.crf_layer.hidden2tag(output)

            # get the lable score ( if the label is unk get averge of all label scores
            y_char = y_char % self.tagset_size
            tag_scores = []
            # @ todo cuda
            for i, y in enumerate(y_char):
                if y == self.o_tag and self.partial == True:
                    tag_scores.append(torch.mean(this_representation_o_t[:, i, :]))
                else:
                    tag_scores.append(this_representation_o_t[:, i, y].squeeze(0))
            tag_scores = torch.stack(tag_scores).unsqueeze(0)
            rep = torch.cat([w_last_node, tag_scores], dim=-1)
        return rep

    def test(self, val_loader):
        """"""
        self.tagger_model.eval()
        total_loss = 0
        predictions = []
        sents_decoded = []
        actual_tags = []

        for i, (sents, tags, tags_iobes, signs, s_lengths, _,_) in enumerate(val_loader):
            max_word_len = max(s_lengths.tolist())

            if self.args.iobes:
                tags = tags_iobes


            # Reduce batch's padded length to maximum in-batch sequence
            # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
            sents = sents[:, :max_word_len]
            tags = tags[:, :max_word_len]
            s_lengths = s_lengths.to(device)
            loss, decoded, w_lengths, tags_sorted, = self.tagger_model.forward_eval(sents, s_lengths,tags)

            decoded = decoded.transpose(1, 0)
            decoded_sorted = [decoded[:length-1] for decoded, length in zip(decoded, w_lengths)]
            tags_sorted = [tag[:length-1] for tag, length in zip(tags_sorted, w_lengths)]
            sents_sorted = [sent[:length-1] for sent, length in zip(sents, w_lengths)]
            sents_decoded.extend(sents_sorted)
            predictions.extend(decoded_sorted)

            total_loss += float(loss)
            actual_tags.extend(tags_sorted)

        total_loss = float(total_loss / len(val_loader))

        return total_loss, predictions, sents_decoded, actual_tags
class Trainer_PA(object):
    """"""

    def __init__(self, args, tagger_mdl, optimizer_tagger, partial=False):
        """"""
        self.args = args
        self.tagger_model = tagger_mdl


        self.tagset_size = tagger_mdl.tagset_size
        self.o_tag = tagger_mdl.tags_vocab['O']
        self.partial=partial

        self.optimizer_tagger = optimizer_tagger
        self.epoch = 0

    def train(self, train_loader, epoch):

        """"""
        self.tagger_model.train()

        self.epoch = epoch
        total_loss = 0.0

        PA_s=0
        for data in tqdm(train_loader, desc='Tagger optimizing  epoch  ' + str(self.epoch) + ''):
            sents, tags, tags_iobes, signs, s_lengths, tags_one_hot, tags_iobes_one_hot=data
            max_word_len = max(s_lengths.tolist())
            if self.args.iobes:
                tags= tags_iobes
                tags_one_hot=tags_iobes_one_hot

            tags = tags[:, :max_word_len].to(device)
            tags_one_hot=tags_one_hot[:, :max_word_len].byte().to(device)
            batch_loss = self.tagger_model(sents, s_lengths,tags_one_hot)

            total_loss += batch_loss.item()
            self.optimizer_tagger.zero_grad()
            batch_loss.backward()
            self.optimizer_tagger.step()

            #print ('total loss: '+ str(total_loss))
        total_loss= float(total_loss / (len(train_loader)))
        print("epoch {} end, train loss:{} , pa-instances: {} " .format(epoch, total_loss ,PA_s))
        return total_loss

    def getState(self, trn_loader, x_unls, x_candidate, y_candidate):

        x_trns=[]
        y_trns=[]
        y_iobes_trns=[]
        x_lens=[]
        self.tagger_model.eval()

        with torch.no_grad():
            for i_batch, (x_trn, y_trn, y_iobes_trn,_, x_len) in enumerate(trn_loader):
                x_trns.extend(x_trn)
                y_trns.extend(y_trn)
                y_iobes_trns.extend(y_iobes_trn)
                x_lens.extend(x_len)
            if self.args.iobes:
                y_trn = y_iobes_trns

            #@todo cuda device
            x_trns=torch.stack(x_trns)
            y_trns=torch.stack(y_trns)
            x_lens =torch.stack(x_lens)

            w_emb = self.tagger_model.embeddings(x_trns)

            sum_w_emb=torch.sum(torch.sum(w_emb, dim=0), dim=1)
            # @todo get sum of embedding based on the len each sentence
            # lstm
            w, _ = self.tagger_model.BiLSTM(w_emb)

            this_representation_o_t = self.tagger_model.MLP(w)
            # get the lable score ( if the label is unk get averge of all label scores
            y_chars = y_trns % self.tagset_size
            tag_scores = []
            for i, y in enumerate(y_chars):
                tag_score=[]
                tag_score=torch.stack([this_representation_o_t[i, j, y_i] for j, y_i in enumerate(y)])
                tag_scores.append(tag_score)

            # @todo get scores based on the len each sentence
            tag_scores = torch.stack(tag_scores)
            sum_tag_scores=torch.sum(tag_scores,dim=0)

            unls_emb = self.tagger_model.embeddings(x_unls)
            sum_unls_emb=torch.sum(torch.sum(unls_emb, dim=0), dim=1)

            # fake dim

            em_candidate=self.tagger_model.embeddings(x_candidate).unsqueeze(0)
            sum_em_candidate = torch.sum(torch.sum(em_candidate, dim=0), dim=1)
            w_candidate, _ = self.tagger_model.BiLSTM(em_candidate)

            candidate_o_t = self.tagger_model.MLP(w_candidate)
            # get the lable score ( if the label is unk get averge of all label scores
            y_char = y_candidate  % self.tagset_size
            tag_scores = []

            tag_scores_candidate =torch.stack([candidate_o_t[:, i, y].squeeze(0) for  i, y in enumerate(y_char)])

            state = torch.cat([sum_w_emb, sum_tag_scores, sum_unls_emb, sum_em_candidate, tag_scores_candidate ], dim=-1)

        return state

    def test(self, val_loader):
        """"""
        self.tagger_model.eval()
        total_loss = 0
        predictions = []
        sents_decoded = []
        actual_tags = []

        for i, (sents, tags, tags_iobes, signs, s_lengths,tags_one_hot,tags_iobes_one_hot) in enumerate(val_loader):
            max_word_len = max(s_lengths.tolist())

            if self.args.iobes:
                tags = tags_iobes
                tags_one_hot=tags_iobes_one_hot

            # Reduce batch's padded length to maximum in-batch sequence
            # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
            sents = sents[:, :max_word_len].to(device)
            tags = tags[:, :max_word_len].to(device)

            s_lengths = s_lengths.to(device)
            loss, decoded, w_lengths, tags_sorted, word_sort_ind= self.tagger_model.forward_eval(sents, s_lengths,tags)

            decoded=decoded.transpose(1,0)
            decoded_sorted = [decoded[:length-1] for decoded, length in zip(decoded, w_lengths)]
            # Remove timesteps we won't predict at, and also <end> tags, because to predict them would be cheating
            tags_sorted=[tag[:length-1] for tag, length in zip(tags_sorted, w_lengths)]
            sents_sorted=sents[word_sort_ind]
            sents_sorted = [sent[:length-1] for sent, length in zip(sents_sorted, w_lengths)]

            sents_decoded.extend(sents_sorted)
            predictions.extend(decoded_sorted)
            total_loss += loss
            actual_tags.extend(tags_sorted)
        total_loss = float(total_loss / len(val_loader))
        return total_loss, predictions, sents_decoded, actual_tags



class MYTrainer_PA_SL(object):
    """"""
    def __init__(self, args, tagger_mdl, sl_mdl, optimizer_tagger, optimizer_sl, criterion_sl, partial):
        """"""
        self.args = args
        self.tagger_model = tagger_mdl
        self.sl_model = sl_mdl
        self.criterion_sl = criterion_sl
        self.tagset_size = tagger_mdl.tagset_size
        self.o_tag = tagger_mdl.tags_vocab['O']
        self.pad_tag = tagger_mdl.tags_vocab[C.PAD_WORD]

        '''
        it need to define two optimizer:
            1- for selector
            2- for encoder+crf

        first, calculate the reward from encoder+crf
        then,  calculate loss for the selector
        and,  add reward to the loss and optimize the select parameters 
        for name, param in self.model.named_parameters():
                    print (f'{name}: {param.requires_grad}')
        '''
        #

        self.optimizer_tagger = optimizer_tagger
        self.optimizer_sl = optimizer_sl
        self.epoch = 0
        self.partial = partial

    def train(self, dataset, epoch):
        """"""
        self.tagger_model.train()
        # self.optimizer_tagger.zero_grad()

        self.epoch = epoch
        total_loss = 0.0

        cnt = 0
        exist_num_in_batch = 0  # instance num in this batch(expert + select_pa_sample)
        PA_num_in_batch = 0  # select_pa_sample_num in this sample (number of PAs)
        y_label_in_batch = []  # all_pa_data action (0/1)
        PA_sentense_representation = []  # representation of every PA instance
        X_char_batch = []
        y_batch = []
        # sign_batch=[]
        s_length_batch = []
        y_one_hot_batch = []

        # not update
        total_PA_num = 0
        X_char_all = []
        y_all = []
        # sign_all=[]
        s_length_all = []
        y_one_hot_all = []

        indices = torch.randperm(len(dataset))

        for start_index in tqdm(range(len(dataset)), desc='Training epoch  ' + str(self.epoch) + ''):

            sent, tags, tags_iobes, sign, s_length, y_one_hot, y_iobes_one_hot = dataset[indices[start_index]]

            if self.args.iobes:
                tags = tags_iobes
                y_one_hot = y_iobes_one_hot
                del y_iobes_one_hot, tags_iobes
            else:
                del y_iobes_one_hot, tags_iobes


            if exist_num_in_batch == self.args.batch_size:  # if the number of the one that selected + experts== batch_size
                X_char_all.extend(X_char_batch)
                y_all.extend(y_batch)
                s_length_all.extend(s_length_batch)
                # sign_all.extend(sign_batch)
                y_one_hot_all.extend(y_one_hot_batch)
                cnt += 1
                '''
                    optimize the selector:
                    1. count reward: add all p(y|x) of dev_dataset and average
                    2. input1: average_reward
                    3. input2: all PA_sample in this step(0 or 1)
                '''
                if len(y_label_in_batch) > 0:
                    # calculate reward as r=1/(|A_i| +|H_i|) *(sum(log p(z|x)) + sum(log p(y|x)) just for EXperts and PA that selector choose
                    reward = self.get_reward(X_char_batch, s_length_batch,y_batch, y_one_hot_batch)
                    reward_list = [reward for i in range(len(y_label_in_batch))]

                    # how to model state : s_i: PA_sentense_representation,  s_(i-1): ?
                    self.optimize_selector(PA_sentense_representation, y_label_in_batch, reward_list)

                total_PA_num += PA_num_in_batch
                exist_num_in_batch = 0
                PA_num_in_batch = 0
                y_label_in_batch = []
                PA_sentense_representation = []
                X_char_batch = []
                y_batch = []
                sign_batch = []
                s_length_batch = []
                y_one_hot_batch = []

            if sign == 0:  # if it is the expert instance
                exist_num_in_batch += 1
                X_char_batch.append(sent)
                y_batch.append(tags)
                # sign_batch.append(sign)
                s_length_batch.append(s_length)
                y_one_hot_batch.append(y_one_hot)

            elif sign == 1:  # the PA instance
                this_representation = self.get_representation(sent, tags, s_length)
                PA_sentense_representation.append(this_representation)
                action_point = self.select_action(
                    this_representation)  # Get the probablity fro selector for the sentence
                if action_point > 0.5:
                    X_char_batch.append(sent)
                    y_batch.append(tags)
                    # sign_batch.append(sign)
                    s_length_batch.append(s_length)
                    y_one_hot_batch.append(y_one_hot)
                    PA_num_in_batch += 1
                    exist_num_in_batch += 1
                    y_label_in_batch.append(1)
                else:
                    y_label_in_batch.append(0)

        if exist_num_in_batch <= self.args.batch_size and exist_num_in_batch > 0:
            cnt += 1
            left_size = self.args.batch_size - exist_num_in_batch
            for i in range(left_size):
                index = np.random.randint(len(dataset))
                sent, tags, tags_iobes, sign, s_length, y_one_hot, y_one_hot_iobes = dataset[index]
                if self.args.iobes:
                    tags = tags_iobes
                    y_one_hot=y_one_hot_iobes
                X_char_batch.append(sent)
                y_batch.append(tags)
                sign_batch.append(sign)
                s_length_batch.append(s_length)
                y_one_hot_batch.append(y_one_hot)
            X_char_all.extend(X_char_batch)
            y_all.extend(y_batch)
            # sign_all.extend(sign_batch)
            s_length_all.extend(s_length_batch)
            y_one_hot_all.extend(y_one_hot_batch)

            if len(y_label_in_batch) > 0:
                reward = self.get_reward(X_char_batch, s_length_batch, y_batch, y_one_hot_batch)
                reward_list = [reward for i in range(len(y_label_in_batch))]
                # just for PA (0,1)
                self.optimize_selector(PA_sentense_representation, y_label_in_batch, reward_list)


        # optimize baseline
        num_iterations = int(math.ceil(1.0 * len(X_char_all) / self.args.batch_size))
        total_loss = 0

        for iteration in tqdm(range(num_iterations), desc='Tagger optimizing  epoch  ' + str(self.epoch) + ''):
            self.tagger_model.train()
            self.optimizer_tagger.zero_grad()
            X_char_train_batch, s_length_batch, y_one_hot_batch = nextBatch(X_char_all, s_length_all, y_one_hot_all,
                                                                                           start_index=iteration * self.args.batch_size,
                                                                                           batch_size=self.args.batch_size)

            X_char_train_batch = torch.stack(X_char_train_batch)
            #y_train_batch = torch.stack(y_train_batch)
            s_length_batch = torch.stack(s_length_batch)
            y_one_hot_batch = torch.stack(y_one_hot_batch)

            batch_loss = self.tagger_model(X_char_train_batch, s_length_batch, y_one_hot_batch)

            total_loss += float(batch_loss)
            # retain_graph=True
            batch_loss.backward()
            self.optimizer_tagger.step()

        total_loss = float(total_loss / num_iterations)
        cnt += 1
        print("epoch %d iteration %d end, train_PA loss: %.4f, total_PA_num: %5d" % (
        epoch, cnt, total_loss, total_PA_num))
        # Decay learning rate every epoch
        return total_loss

    def get_reward(self, x_words, lengths,y_tags, y_tags_one_hot,F1_previous=0):
        self.tagger_model.eval()
        with torch.no_grad():
            x_words = torch.stack(x_words)
            lengths = torch.stack(lengths)
            #y_tags=torch.stack(y_tags)
            y_tags_one_hot = torch.stack(y_tags_one_hot)
            batch_loss = self.tagger_model(x_words, lengths, y_tags_one_hot)
            reward = -1 * (batch_loss / self.args.batch_size)
            # pr, recall, f1
            preds, w_lengths, tmaps, word_sort_ind= self.tagger_model.forward_sl(x_words, lengths, y_tags)
            actual_tags_2_label, pred_2_label=tags_to_labels(y_tags, preds, self.tagger_mdl.tags_vocab,
                                   self.args.iobes)
            prec, rec, f1 = evaluate(actual_tags_2_label, actual_tags_2_label, verbose=False)

        return reward,f1

    def select_action(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0)
        self.sl_model.eval()
        prob = self.sl_model(state)

        return prob.item()

    def optimize_selector(self, x_representations, y_select, rewards):
        self.sl_model.train()
        self.optimizer_sl.zero_grad()
        #eps = np.finfo(np.float32).eps.item()

        x_representations = torch.stack(x_representations).to(device)
        y_select = torch.FloatTensor(y_select).to(device)

        y_preds = self.sl_model(x_representations)
        y_preds = y_preds.squeeze(2).squeeze(1)

        neg_log_prob = self.criterion_sl(y_preds, y_select)
        rewards = torch.FloatTensor(rewards).to(device)
        policy_loss = torch.sum(neg_log_prob * rewards)
        policy_loss = neg_log_prob * rewards

        lambda1, lambda2 = 0.003, 0.003
        all_linear1_params = torch.cat([x.view(-1) for x in self.sl_model.affine1.parameters()])
        all_linear2_params = torch.cat([x.view(-1) for x in self.sl_model.affine1.parameters()])
        l1_regularization = lambda1 * torch.norm(all_linear1_params, 1)
        l2_regularization = lambda2 * torch.norm(all_linear2_params, 2)
        policy_loss += l1_regularization + l2_regularization
        torch.autograd.backward(policy_loss,grad_tensors=torch.ones_like(policy_loss))
        #policy_loss.backward()
        self.optimizer_sl.step()

    def get_representation(self, X_char, y_char, length):
        # concat sample_representation and tag_representation
        # Get representation from BiLSTM for the current sentence
        with torch.no_grad():
            X_char = X_char.to(device)
            w = self.tagger_model.embeddings(X_char).unsqueeze(0)
            w = self.tagger_model.dropout(w)

            # Pack padded sequence
            w = torch.nn.utils.rnn.pack_padded_sequence(w, list([self.args.max_len]),
                                                        batch_first=True)
            # lstm
            output, (h, cell) = self.tagger_model.BiLSTM(
                w)  # packed sequence of word_rnn_dim, with real sequence lengths

            # Unpack packed sequence
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                               batch_first=True)  # (batch_size, max_word_len_in_batch, word_rnn_dim)

            output = self.tagger_model.dropout(output)

            w_last_node = output[:, length - 1, :]

            this_representation_o_t = self.tagger_model.crf_layer.hidden2tag(output)

            # get the lable score ( if the label is unk get averge of all label scores
            y_char = y_char % self.tagset_size
            tag_scores = []
            # @ todo cuda
            for i, y in enumerate(y_char):
                if y == self.o_tag and self.partial == True:
                    tag_scores.append(torch.mean(this_representation_o_t[:, i, :]))
                else:
                    tag_scores.append(this_representation_o_t[:, i, y].squeeze(0))
            tag_scores = torch.stack(tag_scores).unsqueeze(0)
            rep = torch.cat([w_last_node, tag_scores], dim=-1)
        return rep

    def test(self, val_loader):
        """"""
        self.tagger_model.eval()
        total_loss = 0
        predictions = []
        sents_decoded = []
        actual_tags = []

        for i, (sents, tags, tags_iobes, signs, s_lengths, _,_) in enumerate(val_loader):
            max_word_len = max(s_lengths.tolist())

            if self.args.iobes:
                tags = tags_iobes


            # Reduce batch's padded length to maximum in-batch sequence
            # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
            sents = sents[:, :max_word_len]
            tags = tags[:, :max_word_len]
            s_lengths = s_lengths.to(device)
            loss, decoded, w_lengths, tags_sorted, = self.tagger_model.forward_eval(sents, s_lengths,tags)

            decoded = decoded.transpose(1, 0)
            decoded_sorted = [decoded[:length-1] for decoded, length in zip(decoded, w_lengths)]
            tags_sorted = [tag[:length-1] for tag, length in zip(tags_sorted, w_lengths)]
            sents_sorted = [sent[:length-1] for sent, length in zip(sents, w_lengths)]
            sents_decoded.extend(sents_sorted)
            predictions.extend(decoded_sorted)

            total_loss += float(loss)
            actual_tags.extend(tags_sorted)

        total_loss = float(total_loss / len(val_loader))

        return total_loss, predictions, sents_decoded, actual_tags


class trainer_policy(object):
    """"""
    def __init__(self, args, policy_model, policy_optimizer):
        """"""
        self.args = args
        self.policy_model = policy_model
        self.policy_optimizer = policy_optimizer


    def train(self, train_loader, epoch):
        """"""
        self.policy_model.train()

        self.epoch = epoch
        total_loss = 0.0
        self.policy_optimizer.zero_grade()

        for states, actions_actual, in train_loader:
            self.policy_optimizer.zero_grad()
            p_action= self.policy_model(states)
            loss = F.nll_loss(p_action, actions_actual)
            # backward
            total_loss += loss
            loss.backward()
            total_loss+=loss
            self.policy_optimizer.step()
        total_loss=float(total_loss/len(train_loader))
        return total_loss


    def test(self,val_loader):
        self.policy_model.eval()
        predictions = []
        for i, (x, y) in enumerate(
                val_loader):
            p_actions= self.policy_model(x,y)
            preds= p_actions.data.max(1, keepdim=True)[1]
            predictions.extend(preds)
        return predictions




def nextBatch(X_char, s_lengths,y_one_hots, start_index, batch_size=128):
    last_index = start_index + batch_size
    X_char_batch = list(X_char[start_index:min(last_index, len(X_char))])
    #y_batch = list(y[start_index:min(last_index, len(X_char))])
    s_lengths_batch=list(s_lengths[start_index:min(last_index, len(s_lengths))])
    y_one_hots_batch = list(y_one_hots[start_index:min(last_index, len(y_one_hots))])

    if last_index > len(X_char):
        left_size = last_index - (len(X_char))
        for i in range(left_size):
            index = np.random.randint(len(X_char))
            X_char_batch.append(X_char[index])
            #y_batch.append(y[index])
            s_lengths_batch.append((s_lengths[index]))
            y_one_hots_batch.append((y_one_hots[index]))


    return X_char_batch, s_lengths_batch, y_one_hots_batch


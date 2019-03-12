
import os
import sys
# necessary to add cwd to path when script run
# by slurm (since it executes a copy)
sys.path.append(os.getcwd())

import random
import argparse
import logging
import torch
import torch.optim as optim
import constants as C
from vocab import Vocab
from utils import load_word_vectors_EC, build_vocab, write_output
from dataset import EC_PA_Datset
from trainer import  Trainer_PA_SL_DSNER, Trainer_PA
from model import  BiLSTM_CRF_PA_CRF_LightNER, policy_selector
from conlleval import evaluate, tags_to_labels
from weight_init import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch implementation of Distantly Supervised NER with Partial Annotation Learning and Reinforcement Learning ")
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    parser.add_argument('--data', default=f'{data_dir}/EC/',
                        help='path to dataset')
    parser.add_argument('--glove', default=f'{data_dir}/EC/embeddings/',
                        help='directory with glove word embeddings')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--expname', type=str, default='test',
                        help='Name to identify experiment')

    parser.add_argument('--cased', type=bool, default=False,
                        help='if we consider lower cased true of surface form false')

    parser.add_argument('--max_len', default=75, type=int,
                        help="Max lenght of input")
    parser.add_argument('--word_size', default=100, type=int,
                        help="word embedding size (default: 100)")

    parser.add_argument('--freeze_embed', action='store_true', default=False,
                        help='Freeze word embeddings')

    parser.add_argument('--batch_size', default=64, type=int,
                        help="batchsize for optimizer updates (default: 25)")

    parser.add_argument('--epochs', default=800, type=int,
                        help="number of total epochs to run (default: 10)")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="initial learning rate (default: 0.001)")
    parser.add_argument('--weight_decay', default=1e-4, type=int,
                        help="weight decay (default: 0.0001)")
    parser.add_argument('--dropout', default=0.5, type=float,
                        help="use dropout (default: 0.5), -1: no dropout")

    parser.add_argument('--emblr', default=0.1, type=float,
                        help="initial embedding learning rate (default: 0.1)")

    parser.add_argument('--optim', default="adam",
                        help="optimizer (default: adam)")

    parser.add_argument('--seed', default=123, type=int,
                        help="random seed (default: 123)")

    parser.add_argument('--SL_hidden_size', default=100, type=int,
                        help="selector hidden size (default: 100)")

    parser.add_argument('--biLSTM_hidden_size', default=100, type=int,
                        help="word BiLSTM hidden size (default: 200)")

    parser.add_argument('--iobes', default=False, type=bool,
                        help="if to use BIOES tagging style")

    parser.add_argument('--setup', default='H',
                        help="which dataset: H, A+H")

    parser.add_argument('--mode', default='SL',
                        help="Mode of training: PA+SL,PA ,SL, CRF")

    return parser.parse_args()


def main():
    global args
    args = parse_args()
    # logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save):
        os.makedirs(args.save)


    ##print args:
    dataset = 'EC'
    print(f'{args.mode}')

    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')
    ds_pa_dir=os.path.join(args.data, 'ds_pa/')






    ner_vocab_file = os.path.join(args.data, 'ner.vocab')
    if not os.path.isfile(ner_vocab_file):
        token_files = [os.path.join(split, 'a.txt') for split in
                       [train_dir, test_dir, dev_dir, ds_pa_dir] ]#[train_dir, dev_dir, test_dir]
        ner_vocab_file = os.path.join(args.data, 'ner.vocab')
        build_vocab(token_files, ner_vocab_file)

    vocab = Vocab(filename=ner_vocab_file)
    vocab.add_specials([C.UNK_WORD, C.PAD_WORD])


    ner_vocab_file = os.path.join(args.data, 'ner.tags.vocab' )
    if not os.path.isfile(ner_vocab_file):
        tags_files = [os.path.join(split, 'tags.txt') for split in
                       [train_dir, dev_dir, test_dir, ds_pa_dir]]  # [train_dir, dev_dir, test_dir]
        ner_vocab_file = os.path.join(args.data, 'ner.tags.vocab')
        build_vocab(tags_files, ner_vocab_file)

    tags_vocab = Vocab(filename=ner_vocab_file)
    tags_vocab.add_specials([C.BOS_WORD,C.PAD_WORD])

    ner_vocab_file = os.path.join(args.data, 'ner.tags-iobes.vocab')
    if not os.path.isfile(ner_vocab_file):
        tags_files = [os.path.join(split, 'tags-iobes.txt') for split in
                      [train_dir, dev_dir, test_dir, ds_pa_dir]]  # [train_dir, dev_dir, test_dir]
        ner_vocab_file = os.path.join(args.data, 'ner-iobes.tags.vocab')
        build_vocab(tags_files, ner_vocab_file)

    tags_iobes_vocab = Vocab(filename=ner_vocab_file)  #
    #tags_iobes_vocab.add_specials([C.BOS_WORD,C.EOS_WORD])
    tags_iobes_vocab.add_specials([C.BOS_WORD,C.PAD_WORD])


    # load ner dataset splits
    #train_file = os.path.join(args.data, 'ner.train.pth')
    #if os.path.isfile(train_file):
    #    train_dataset = torch.load(train_file)
    #else:



    train_dataset = EC_PA_Datset(train_dir, vocab, tags_vocab, tags_iobes_vocab)
    #torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))


    #test_file = os.path.join(args.data, 'ner.test.pth')
    # if os.path.isfile(test_file):
    #     test_dataset = torch.load(test_file)
    # else:
    test_dataset = EC_PA_Datset(test_dir, vocab, tags_vocab, tags_iobes_vocab)
    #torch.save(test_dataset, test_file)
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    #dev_file = os.path.join(args.data, 'ner.dev.pth')
    # if os.path.isfile(dev_file):
    #     dev_dataset = torch.load(dev_file)
    # else:
    dev_dataset = EC_PA_Datset(dev_dir, vocab, tags_vocab, tags_iobes_vocab)
    #torch.save(dev_dataset, dev_file)
    logger.debug('==> Size of dev data    : %d ' % len(dev_dataset))

    ds_pa_file = os.path.join(args.data, 'ner.ds_pa.pth')
    #if os.path.isfile(ds_pa_file):
     #   ds_pa_dataset = torch.load(ds_pa_file)
    #else:
    ds_pa_dataset = EC_PA_Datset(ds_pa_dir, vocab, tags_vocab, tags_iobes_vocab, partial= True if 'PA' in args.mode else False)
    #torch.save(ds_pa_dataset, ds_pa_file)
    logger.debug('==> Size of ds pa data    : %d ' % len(ds_pa_dataset))

    #merge_file = os.path.join(args.data, 'ner.merge.pth')
    #if os.path.isfile(merge_file):
    #    merge_dataset = torch.load(merge_file)
    #else:
    merge_dataset = ds_pa_dataset.merge(train_dataset)
    #torch.save(merge_dataset, merge_file)
    logger.debug('==> Size of merge  data : %d ' % len(merge_dataset))

    if args.iobes:
        tags_vocab=tags_iobes_vocab


    tagger_model = BiLSTM_CRF_PA_CRF_LightNER(
        vocab=vocab.label_to_idx,
        emb_size=args.word_size,
        tags_vocab=tags_vocab.label_to_idx,
        freeze_embed=True,
        biLSTM_hidden_size=args.biLSTM_hidden_size,
        dropout=args.dropout)

    sl_model= policy_selector(2*args.biLSTM_hidden_size+args.max_len, args.SL_hidden_size)

    tagger_model.apply(weight_init)
    sl_model.apply(weight_init)

    if args.cuda:
        tagger_model=tagger_model.to('cuda')
        sl_model=sl_model.to('cuda')

        '''
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        '''

    optimizer_tagger=optim.RMSprop(filter(lambda p: p.requires_grad, tagger_model.parameters()), lr=args.lr,
                               weight_decay=args.weight_decay)


    criterion_sl = torch.nn.BCELoss(reduction='none')

    if args.optim == 'adam':
        optimizer_sl = optim.Adam(filter(lambda p: p.requires_grad, sl_model.parameters()), lr=args.lr)
    elif args.optim == 'adagrad':
        optimizer_sl = optim.Adagrad(filter(lambda p: p.requires_grad, sl_model.parameters()), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer_sl = optim.SGD(filter(lambda p: p.requires_grad, sl_model.parameters()), lr=args.lr,
                              weight_decay=args.weight_decay)


    # word embedding
    emb_file = os.path.join(args.data, 'ner.embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors_EC(os.path.join(args.glove, 'pre_trained_100dim.model'))
        logger.debug('==> EMBEDDINGS vocabulary size: %d ' % glove_vocab.size())
        emb = torch.Tensor(vocab.size(), glove_emb.size(1)).uniform_(-0.05, 0.05)
        # zero out the embeddings for padding and other special words
        for idx, item in enumerate([C.PAD_WORD, C.UNK_WORD]):
            emb[idx].zero_()
        for word in vocab.label_to_idx.keys():
            if glove_vocab.get_index(word):
                emb[vocab.get_index(word)] = glove_emb[glove_vocab.get_index(word)]
        torch.save(emb, emb_file)

    if args.cuda:
        emb = emb.cuda()

    tagger_model.embeddings.weight.data.copy_(emb)


    if args.mode=='PA+SL':
        # if partial = False it will apply only selection with normal crf
        trainer = Trainer_PA_SL_DSNER(args, tagger_model, sl_model, optimizer_tagger, optimizer_sl, criterion_sl, partial=True)
    else:
        # the result of partial = False or partial =True should be same because we do not include the Partial annotation
        trainer = Trainer_PA(args, tagger_model, optimizer_tagger, partial=False)

    best = -float('inf')



    # dataset:
    if args.setup=='A+H':
        dataset_setup= merge_dataset
    else:
        dataset_setup = train_dataset

    #print(tagger_model)

    f1s={}
    for epoch in range(args.epochs):

        if args.mode == 'PA+SL':
            train_loss = trainer.train(dataset_setup, epoch)
            train_f1 = 'NAN'

        else:
            train_loader = torch.utils.data.DataLoader(dataset_setup, batch_size=args.batch_size, shuffle=True,
                                                      num_workers =1, pin_memory=False)
            _=trainer.train(train_loader,epoch)

            train_loader_test = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=1, pin_memory=False)

            train_loss, train_pred, train_sent ,actual_tags = trainer.test(train_loader_test)
            actual_tags = [[int(t) for t in tags.cpu().numpy()] for tags in actual_tags]
            train_pred = [[int(t) for t in tags.cpu().numpy()] for tags in train_pred]

           # using conlleval.py
            actual_tags_2_label, train_pred_2_label=tags_to_labels(actual_tags, train_pred, tags_vocab.idx_to_label,
                                   args.iobes)

            prec, rec, train_f1 = evaluate(actual_tags_2_label,train_pred_2_label, verbose=False)
            train_f1 = round(train_f1, 2)

        print('==> Epoch {}, Train \tLoss: {:.2f}\tf1: {}'.format(epoch + 1, train_loss, train_f1))

        if args.cuda:
            torch.cuda.empty_cache()

        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=1, pin_memory=False)
        dev_loss, dev_pred,  dev_actual_tags = trainer.test(dev_loader)

        dev_actual_tags = [[int(t) for t in tags.cpu().numpy()] for tags in dev_actual_tags]
        dev_pred = [[int(t) for t in tags.cpu()] for tags in dev_pred]


        dev_actual_tags_2_label, dev_train_pred_2_label = tags_to_labels(dev_actual_tags, dev_pred,
                                                                            tags_vocab.idx_to_label,
                                                                            args.iobes)
        prec, rec, dev_f1 = evaluate(dev_actual_tags_2_label, dev_train_pred_2_label, verbose=False)
        print('==> Epoch {}, dev \tLoss: {:.2f}\tf1: {:.2f}'.format(epoch + 1, dev_loss, dev_f1))


        if best < dev_f1:
            best = dev_f1

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=1, pin_memory=False)

            test_loss, test_pred,  test_actual_tags = trainer.test(test_loader)

            test_actual_tags = [[int(t) for t in tags.cpu().numpy()] for tags in test_actual_tags]
            test_pred = [[int(t) for t in tags.cpu()] for tags in test_pred]

            test_actual_tags_2_label, test_train_pred_2_label = tags_to_labels(test_actual_tags, test_pred,
                                                                             tags_vocab.idx_to_label,
                                                                             args.iobes)

            prec, rec, test_f1 = evaluate(test_actual_tags_2_label, test_train_pred_2_label, verbose=False)

            print('==> Epoch {}, Test \tLoss: {:.2f}\tf1: {:.2f}'.format(epoch + 1, test_loss, test_f1))

            checkpoint = {
                'model_tagger': trainer.tagger_model.state_dict(),
                'optim_tagger': trainer.optimizer_tagger,
                'f1': test_f1,
                'f1-tags': test_f1,
                'predict': test_pred,
                'args': args,
                'epoch': epoch + 1
            }

            dataset = 'ER'
            filename = 'ner-{}-{}-{}-{}-{}'.format(dataset, args.mode,args.setup,
                                                                 args.batch_size,
                                                                 args.seed)
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, filename))

            f1s[epoch] = {'T': [round(train_f1,2)], 'D': [round(dev_f1,2)], 'E': [round(test_f1,2)]}
            print(f1s)

    out_name = 'ner-{}-{}-{}'.format(dataset, args.mode, args.setup)
    output_file = os.path.join(args.data, out_name)
    write_output(output_file,f1s)






if __name__ == "__main__":


    main()

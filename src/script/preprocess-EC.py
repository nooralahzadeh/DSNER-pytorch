"""
Preprocessing script for EC data.

"""

import os
from src import utils
def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)



def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')



def read_split_datafiles(filepath, dst_dir, tag_ind, caseless=False):
    """
       Reads raw data
       :return: word, tag sequences
       """
    cased='lower' if caseless else ''
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'a%s.txt' % cased), 'w') as afile, \
         open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
         open(os.path.join(dst_dir, 'tags.txt'), 'w') as tagsfile, \
         open(os.path.join(dst_dir, 'tags-iobes.txt'), 'w') as tagsfile_iobes:
            index=0
            temp_w = []
            temp_t = []
            idx=0
            for line in datafile:
                idx+=1
                if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
                    feats = line.rstrip('\n').split("\t")
                    print (f'{feats}-{idx}')
                    # convert to digit
                    token=feats[0].lower() if caseless else feats[0]
                    #token=re.sub('\d', '0', token)
                    temp_w.append(token)
                    temp_t.append(feats[tag_ind])
                elif len(temp_w) > 0:
                    assert len(temp_w) == len(temp_t)
                    idfile.write(str(index) + '\n')
                    afile.write('\t'.join(temp_w)+ '\n')
                    tagsfile.write('\t'.join(temp_t) + '\n')
                    tagsfile_iobes.write('\t'.join(utils.iob_iobes(temp_t))+ '\n')
                    temp_w = []
                    temp_t = []
                    index += 1
            # last sentence
            if len(temp_w) > 0:
                assert len(temp_w) == len(temp_t)
                idfile.write(str(index) + '\n')
                afile.write('\t'.join(temp_w) + '\n')
                tagsfile.write('\t'.join(temp_t) + '\n')
                tagsfile_iobes.write('\t'.join(utils.iob_iobes(temp_t))+ '\n')




if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing EC PA  dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    ner_dir = os.path.join(data_dir, 'EC')
    train_dir = os.path.join(ner_dir, 'train')
    dev_dir = os.path.join(ner_dir, 'dev')
    test_dir = os.path.join(ner_dir, 'test')
    ds_pa=os.path.join(ner_dir, 'ds_pa')

    make_dirs([train_dir, dev_dir, test_dir,ds_pa])



    # split into separate file
    caseless=False
    cased = 'lower' if caseless else ''
    read_split_datafiles(os.path.join(ner_dir, 'train.txt'), train_dir,tag_ind=1, caseless=caseless)
    read_split_datafiles(os.path.join(ner_dir, 'dev.txt'), dev_dir,tag_ind=1,caseless=caseless)

    read_split_datafiles(os.path.join(ner_dir, 'test.txt'), test_dir,tag_ind=1,caseless=caseless)
    read_split_datafiles(os.path.join(ner_dir, 'ds_fa.txt'), ds_pa, tag_ind=1,caseless=caseless)







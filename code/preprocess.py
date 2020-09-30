import logging
import torch
import seq2seq

try:
    import ipdb
except ImportError:
    pass

lower = True
seq_length = 200
report_every = 10000
shuffle = 1

logger = logging.getLogger(__name__)


def makeVocabulary(filenames, size):
    vocab = seq2seq.Dict([seq2seq.Constants.PAD_WORD, seq2seq.Constants.UNK_WORD,
                      seq2seq.Constants.BOS_WORD, seq2seq.Constants.EOS_WORD], lower=lower)
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().split(' '):
                    vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    logger.info('Created dictionary of size %d (pruned from %d)' %
                (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFiles, vocabFile, vocabSize):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        logger.info('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = seq2seq.Dict(lower=lower)
        vocab.loadFile(vocabFile)
        logger.info('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        logger.info('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)

        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):
    logger.info('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts, guideFile, guideDicts):
    src, guide, tgt = [], [], []
    switch, c_tgt = [], []
    sizes = []
    count, ignored = 0, 0

    logger.info('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf-8')
    guideF = open(guideFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')

    while True:
        sline = srcF.readline()
        gline = guideF.readline()
        tline = tgtF.readline()

        # normal end of file (last line == '')
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            logger.info('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        gline = gline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            logger.info('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        # src, tgt are all in lists
        srcWords = sline.split()
        guideWords = gline.split()
        tgtWords = tline.split()

        if len(srcWords) <= seq_length and len(tgtWords) <= seq_length:
            src += [srcDicts.convertToIdx(srcWords, seq2seq.Constants.UNK_WORD)]
            guide += [guideDicts.convertToIdx(guideWords, seq2seq.Constants.UNK_WORD)]
            tgt += [tgtDicts.convertToIdx(tgtWords,seq2seq.Constants.UNK_WORD,
                                          seq2seq.Constants.BOS_WORD)]
            switch_buf = [0] * (len(tgtWords) + 1)
            c_tgt_buf = [0] * (len(tgtWords) + 1)
            for idx, tgt_word in enumerate(tgtWords):
                word_id = tgtDicts.lookup(tgt_word, None)
                if word_id is None:
                    if tgt_word in srcWords:
                        copy_position = srcWords.index(tgt_word)
                        switch_buf[idx + 1] = 1
                        c_tgt_buf[idx + 1] = copy_position
            switch.append(torch.FloatTensor(switch_buf))
            c_tgt.append(torch.LongTensor(c_tgt_buf))

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % report_every == 0:
            logger.info('... %d sentences prepared' % count)

    srcF.close()
    guideF.close()
    tgtF.close()

    if shuffle == 1:
        logger.info('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        guide = [guide[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        switch = [switch[idx] for idx in perm]
        c_tgt = [c_tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    logger.info('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]
    switch = [switch[idx] for idx in perm]
    c_tgt = [c_tgt[idx] for idx in perm]
    guide = [guide[idx] for idx in perm]

    logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
                (len(src), ignored, seq_length))
    return src, tgt, switch, c_tgt, guide


def prepare_data_online(train_src, src_vocab, train_tgt, tgt_vocab, train_guide_src, guide_src_vocab):

    dicts = {}
    dicts['src'] = initVocabulary('source', [train_src], src_vocab, 0)
    dicts['tgt'] = initVocabulary('target', [train_tgt], tgt_vocab, 0)
    dicts['guide_src'] = initVocabulary('guide_source', [train_guide_src], guide_src_vocab, 0)

    logger.info('Preparing training ...')
    train = {}


    train['src'], train['tgt'], train['switch'], train['c_tgt'], train['guide_src'] = makeData(train_src, train_tgt,
                                                             dicts['src'], dicts['tgt'], train_guide_src, dicts['guide_src'])

    dataset = {'dicts': dicts,
               'train': train,
               # 'valid': valid
               }


    return dataset

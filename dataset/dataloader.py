import sys
import os
from collections import Counter

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)

def dataclean(srcData, tgtData, level):

    srcWords, tgtWords, words = [], [], []

    for line in srcData:

        word_list = line.split()
        for word in word_list:
            if word != '<ss>' and ('$' not in word and word.isupper()==False):
                words.append(word)
        srcWords.append(word_list)

    for line in tgtData:
        word_list = line.split()
        if level == 'type':
            for word in word_list:
                if word != '<ss>' and ('$' not in word and word.isupper()==False):
                    words.append(word)
        tgtWords.append(word_list)

    return srcWords, tgtWords, words


for dataset in ['gig_','nyt_']:
    for level in ['type', 'mention']:

        train_src = open('./data/' + dataset + 'shuffled_' + level + '_src_train.txt', encoding='utf-8').readlines()
        train_tgt = open('./data/' + dataset + 'shuffled_' + level + '_tgt_train.txt', encoding='utf-8').readlines()
        valid_src = open('./data/' + dataset + 'shuffled_' + level + '_src_dev.txt', encoding='utf-8').readlines()
        valid_tgt = open('./data/' + dataset + 'shuffled_' + level + '_tgt_dev.txt', encoding='utf-8').readlines()
        test_src  = open('./data/' + dataset + 'shuffled_' + level + '_src_test.txt', encoding='utf-8').readlines()
        test_tgt  = open('./data/' + dataset + 'shuffled_' + level + '_tgt_test.txt', encoding='utf-8').readlines()

        trainSrcData = []
        trainTgtData = []
        validSrcData = []
        validTgtData = []
        testSrcData  = []
        testTgtData  = []

        for line in train_src:
            line = line.strip()
            trainSrcData.append(line)

        for line in train_tgt:
            line = line.strip()
            trainTgtData.append(line)

        for line in valid_src:
            line = line.strip()
            validSrcData.append(line)

        for line in valid_tgt:
            line = line.strip()
            validTgtData.append(line)

        for line in test_src:
            line = line.strip()
            testSrcData.append(line)

        for line in test_tgt:
            line = line.strip()
            testTgtData.append(line)

        trainSrcWords, trainTgtWords, trainDict = dataclean(trainSrcData, trainTgtData, level)
        validSrcWords, validTgtWords, validDict = dataclean(validSrcData, validTgtData, level)
        testSrcWords,  testTgtWords,  testDict  = dataclean(testSrcData, testTgtData, level)

        Dict = trainDict
        Dict.extend(validDict)
        Dict.extend(testDict)
        Dict = [(key, value) for key, value in Counter(Dict).items() if value > 0]

        train_dir = CUR_DIR + '/train/'
        dev_dir = CUR_DIR + '/dev/'
        test_dir = CUR_DIR + '/test'

        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        if not os.path.exists(dev_dir):
            os.mkdir(dev_dir)

        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        eWordsFile = open(CUR_DIR + '/train/' + dataset + level + '_train.txt.source.txt', 'w')
        eTargetFile = open(CUR_DIR + '/train/' + dataset + level + '_train.txt.target.txt', 'w')

        aWordsFile = open(CUR_DIR + '/dev/' + dataset + level + '_dev.txt.shuffle.dev.source.txt', 'w')
        aTargetFile = open(CUR_DIR + '/dev/' + dataset + level + '_dev.txt.shuffle.dev.target.txt', 'w')

        tWordsFile = open(CUR_DIR + '/test/' + dataset + level + '_test.txt.shuffle.test.source.txt', 'w')
        tTargetFile = open(CUR_DIR + '/test/' + dataset + level + '_test.txt.shuffle.test.target.txt', 'w')

        wordDict = open(CUR_DIR + '/train/' + dataset + level + '_vocab_src.txt.20k', 'w')

        for trainsrcword, traintgtword in zip(trainSrcWords, trainTgtWords):
            eWordsFile.write('{}\n'.format(' '.join(trainsrcword)))
            eTargetFile.write('{}\n'.format(' '.join(traintgtword)))

        for validsrcword, validtgtword in zip(validSrcWords, validTgtWords):
            aWordsFile.write('{}\n'.format(' '.join(validsrcword)))
            aTargetFile.write('{}\n'.format(' '.join(validtgtword)))

        for testsrcword, testtgtword in zip(testSrcWords, testTgtWords):
            tWordsFile.write('{}\n'.format(' '.join(testsrcword)))
            tTargetFile.write('{}\n'.format(' '.join(testtgtword)))

        if level == 'mention':
            wordDict.write('<blank> 0\n<unk> 1\n<s> 2\n</s> 3\n<ss> 4\n')
            for idx, word in enumerate(Dict):
                wordDict.write('{} {} {}\n'.format(word[0], idx + 5, word[1]))

        else:
            wordDict.write('<blank> 0\n<unk> 1\n<s> 2\n</s> 3\n<ss> 4\nDIGIT 5\nPERSON 6\nORGANIZATION 7\nLOCATION 8\nLOCATIONCOUNTRY 9\nYEAR 10\nMONTH 11\nDIGITRANK 12\nDIGITUNIT 13\nTIMEUNIT 14\nLENGTHUNIT 15\nDAY 16\nWEEKDAY 17\n')
            for idx, word in enumerate(Dict):
                wordDict.write('{} {} {}\n'.format(word[0], idx + 18, word[1]))



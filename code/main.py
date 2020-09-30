from __future__ import division

import seq2seq
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import logging
import random
import numpy as np

try:
    import ipdb
except ImportError:
    pass

from nltk.translate import bleu_score
from seq2seq.xinit import xavier_normal, xavier_uniform
import os
import xargs
torch.backends.cudnn.enabled=False

parser = argparse.ArgumentParser(description='main.py')
xargs.add_data_options(parser)
xargs.add_model_options(parser)
xargs.add_train_options(parser)

opt = parser.parse_args()

logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log.txt'
if opt.log_home:
    log_file_name = os.path.join(opt.log_home, log_file_name)
file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)
logger = logging.getLogger(__name__)

logger.info('My PID is {0}'.format(os.getpid()))
logger.info('PyTorch version: {0}'.format(str(torch.__version__)))
logger.info(opt)

if torch.cuda.is_available() and not opt.gpus:
    logger.info("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.seed > 0:
    torch.manual_seed(opt.seed)

if opt.gpus:
    if opt.cuda_seed > 0:
        torch.cuda.manual_seed(opt.cuda_seed)
    cuda.set_device(opt.gpus[0])

logger.info('My seed is {0}'.format(torch.initial_seed()))
logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))


def Criterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[seq2seq.Constants.PAD] = 0

    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def reverse_Criterion(vocabSize):
    weight = torch.zeros(vocabSize)
    weight[seq2seq.Constants.PAD] = 0

    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def loss_function(g_outputs, g_targets, generator, crit, eval=False):
    batch_size = g_outputs.size(1)

    g_out_t = g_outputs.view(-1, g_outputs.size(2))
    g_prob_t = generator(g_out_t)

    g_loss = crit(g_prob_t, g_targets.view(-1))
    total_loss = g_loss
    report_loss = total_loss.item()
    return total_loss, report_loss, 0


def generate_copy_loss_function(g_outputs, c_outputs, g_targets,
                                c_switch, c_targets, c_gate_values,
                                generator, nlu_generator, classifier, crit, copyCrit,
                                reverse_crit, guideCrit, nluCrit, guides, g_hiddens, nlu_outputs,type_emb):

    # g_out is output [(TargetLength, BatchSize, Hidden)]
    batch_size = g_outputs.size(1)
    targetLength = g_outputs.size(0)

    guides = guides.transpose(1, 0)

    # transform typed target with the special <ent> symbol to mention level target
    m_targets = g_targets.clone().detach()
    m_targets = m_targets.transpose(1, 0)
    for i in range(batch_size):
        count = 0
        for j in range(targetLength):
            if m_targets[i][j] == seq2seq.Constants.SS:
                m_targets[i][j] = guides[i][count]
                count += 1
            else:
                m_targets[i][j] = seq2seq.Constants.PAD

    m_targets = m_targets.transpose(1, 0)

    # calculate nlu prediction distribution
    nlu_out_t = nlu_outputs.view(-1, nlu_outputs.size(2))
    nlu_prob_t = nlu_generator(nlu_out_t)
    nlu_prob_t = nlu_prob_t.view(-1, batch_size, nlu_prob_t.size(1))

    # calculate generation prediction distribution
    g_out_t = g_outputs.view(-1, g_outputs.size(2))
    g_prob_t = generator(g_out_t)
    g_prob_t = g_prob_t.view(-1, batch_size, g_prob_t.size(1))
    g_hiddens = g_hiddens.squeeze(1).transpose(1, 0)

    # calculate the mention predictor loss
    cl_loss = 0
    type_emb = type_emb.transpose(1, 0) # injected type embedding from source
    g_prob_t_trans = g_prob_t.transpose(1, 0)

    g_targets = g_targets.transpose(1, 0)
    guide_probs = []
    for i in range(batch_size):
        guide_prob = []
        guide = []
        count = 0
        for j in range(targetLength):
            if g_targets[i][j] == seq2seq.Constants.SS:
                injected = torch.cat([g_hiddens[i][j],type_emb[i][count]],dim=0)

                guide_prob.append(torch.log(classifier(injected.unsqueeze(0))))
                guide.append(guides[i][count])
                count += 1
        guide_prob = torch.stack(guide_prob).squeeze(1)
        guide = torch.stack(guide).view(-1)
        if i == 0:
            cl_loss = guideCrit(guide_prob,guide)
        else:
            cl_loss += guideCrit(guide_prob,guide)
    g_targets = g_targets.transpose(1, 0)

    g_output_prob = g_prob_t + 1e-8

    g_output_prob_log = torch.log(g_output_prob)
    nlu_prob_log = torch.log(nlu_prob_t)
    g_output_prob_log = g_output_prob_log * ((1 - c_switch).unsqueeze(2).expand_as(g_output_prob_log))

    g_output_prob_log = g_output_prob_log.view(-1, g_output_prob_log.size(2))
    nlu_prob_log = nlu_prob_log.view(-1, nlu_prob_log.size(2))

    # g_output_prob_log [(TargetLength * BatchSize, vocab_size)]
    g_loss = crit(g_output_prob_log, g_targets.view(-1))
    nlu_loss = nluCrit(nlu_prob_log, m_targets.view(-1))

    # 2.5 and 1.5 is hyperparameter
    total_loss = g_loss + cl_loss * 2.5 + nlu_loss * 1.5
    report_loss = total_loss.item()

    return total_loss, report_loss, g_loss, cl_loss, nlu_loss


def addPair(f1, f2, f3):
    for x, y1, z in zip(f1, f2, f3):
        yield (x, y1, z)
    yield (None, None, None)


def load_dev_data(translator, src_file, tgt_file, guide_file):
    dataset, raw = [], []
    srcF = open(src_file, encoding='utf-8')
    guideF = open(guide_file, encoding='utf-8')
    tgtF = open(tgt_file, encoding='utf-8')

    src_batch, tgt_batch = [], []
    guide_batch = []
    for line, tgt, guide in addPair(srcF, tgtF, guideF):
        if (line is not None) and (tgt is not None) and (guide is not None):
            src_tokens = line.strip().split(' ')
            src_batch += [src_tokens]
            guide_tokens = guide.strip().split(' ')
            guide_batch += [guide_tokens]
            tgt_tokens = tgt.strip().split(' ')
            tgt_batch += [tgt_tokens]

            if len(src_batch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(src_batch) == 0:
                break
        data = translator.buildData(src_batch, tgt_batch, guide_batch)
        dataset.append(data)
        raw.append((src_batch, tgt_batch))
        src_batch, tgt_batch = [], []
        guide_batch = []

    srcF.close()
    tgtF.close()
    guideF.close()
    return (dataset, raw)


evalModelCount = 0
totalBatchCount = 0


def evalModel(model, translator, evalData):
    global evalModelCount
    evalModelCount += 1
    ofn_prefix = opt.ofn_prefix
    if evalModelCount % 2 == 0:
        ofn_prefix += '_test'
    else:
        ofn_prefix += '_dev'
    ofn = ofn_prefix + '.{0}'.format(evalModelCount)
    if opt.save_path:
        ofn = os.path.join(opt.save_path, ofn)

    predict, gold = [], []
    processed_data, raw_data = evalData
    for batch, raw_batch in zip(processed_data, raw_data):
        #  (1) read src and tgt
        src, tgt, guide, indices = batch[0]
        src_batch, tgt_batch = raw_batch

        #  (2) translate
        pred, predScore, predIsCopy, predCopyPosition, attn, _ = translator.translateBatch(src, tgt, guide)
        pred, predScore, predIsCopy, predCopyPosition, attn = list(zip(
            *sorted(zip(pred, predScore, predIsCopy, predCopyPosition, attn, indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            n = 0
            predBatch.append(
                translator.buildTargetTokens(pred[b][n], src_batch[b],
                                             predIsCopy[b][n], predCopyPosition[b][n], attn[b][n])
            )

        # nltk BLEU evaluator needs tokenized sentences
        gold += [[r] for r in tgt_batch]
        predict += predBatch
        for i, j in zip([[r] for r in tgt_batch], predBatch):
            if random.random() > 1:
                print('{}\n{}\n{}\n'.format(i, j, '*'*50))

    no_copy_mark_predict = [[word.replace('[[', '').replace(']]', '') for word in sent] for sent in predict]
    bleu = bleu_score.corpus_bleu(gold, no_copy_mark_predict)

    # (4) save generation result
    with open(ofn, 'w', encoding='utf-8') as of:
        for pred in predict:
            of.write(' '.join(pred) + '\n')

    return bleu


def trainModel(model, translator, trainData, validData, testData, dataset, optim):

    logger.info(model)
    model.train()
    torch.cuda.empty_cache()
    for name,para in model.named_parameters():
        if para.requires_grad:
            print(name,para.data.shape)
    logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))

    # define criterion of each GPU
    criterion = Criterion(dataset['dicts']['tgt'].size())
    guideCriterion = Criterion(dataset['dicts']['guide_src'].size())
    nluCriterion = Criterion(dataset['dicts']['guide_src'].size())
    reverse_criterion = reverse_Criterion(dataset['dicts']['tgt'].size())

    copyLossF = nn.NLLLoss(size_average=False)

    start_time = time.time()

    def saveModel(metric=None):
        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(
            opt.gpus) > 1 else model.generator.state_dict()
        #  (5) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }
        save_model_path = 'model'
        if opt.save_path:
            if not os.path.exists(opt.save_path):
                os.makedirs(opt.save_path)
            save_model_path = opt.save_path + os.path.sep + save_model_path
        if metric is not None:
            torch.save(checkpoint, '{0}_dev_metric_{1}_epoch_{2}.pt'.format(save_model_path, round(metric, 4), epoch))
        else:
            torch.save(checkpoint, '{0}_epoch_{1}.pt'.format(save_model_path, epoch))

    def trainEpoch(epoch):
        if opt.extra_shuffle and epoch > opt.curriculum:
            logger.info('Shuffling...')
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
        start = time.time()
        epoch_loss = 0
        epoch_cl_loss = 0
        epoch_nlu_loss = 0
        for i in range(len(trainData)):
            global totalBatchCount
            totalBatchCount += 1

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx][:-1]  # exclude original indices

            model.zero_grad()

            # g_out is output [(TargetLength, BatchSize, Hidden)]
            # c_out is attn_out [(TargetLength, BatchSize, InputLength)]
            # c_gate is copy_prob [(TargetLength, BatchSize, 1)]

            g_outputs, c_outputs, c_gate_values, g_hiddens, nlu_outputs, type_emb = model(input=batch, evaluate=False)

            targets = batch[1][0][1:]  # exclude <SOS> from targets
            copy_switch = batch[1][1][1:]
            c_targets = batch[1][2][1:]
            g_guides = batch[2][0]

            loss, res_loss, g_loss, cl_loss, nlu_loss = generate_copy_loss_function(
                g_outputs, c_outputs, targets, copy_switch, c_targets, c_gate_values, model.generator, model.nlu_generator, model.classifier,
                criterion, copyLossF, reverse_criterion, guideCriterion, nluCriterion, g_guides, g_hiddens, nlu_outputs, type_emb)

            torch.cuda.empty_cache()
            epoch_loss += g_loss
            epoch_cl_loss += cl_loss
            epoch_nlu_loss += nlu_loss

            if math.isnan(res_loss) or res_loss > 1e20:
                logger.info('catch NaN')
                ipdb.set_trace()

            # update the parameters
            loss.backward()
            optim.step()

            num_words = targets.data.ne(seq2seq.Constants.PAD).sum().item()
            report_loss += res_loss
            report_tgt_words += num_words
            report_src_words += batch[0][-1].data.sum()
            total_loss += res_loss
            total_words += num_words
            if i % opt.log_interval == 0:
                logger.info(
                    "Epoch %2d, %1d/%3d; loss: %6.2f; words: %5d; ppl: %6.2f; time: %2.0f sec" %
                    (epoch, totalBatchCount, len(trainData),
                     report_loss,
                     report_tgt_words,
                     math.exp(min((report_loss / report_tgt_words), 16)),
                     time.time() - start))

                report_loss = report_tgt_words = report_src_words = report_num_correct = 0
                start = time.time()

            if validData is not None and totalBatchCount % opt.eval_per_batch == 0 and totalBatchCount >= opt.start_eval_batch:
                model.eval()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
                valid_bleu = evalModel(model, translator, validData)
                model.train()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
                model.decoder.attn.mask = None
                logger.info('Validation Score (bleu): %g' % (valid_bleu * 100))
                if valid_bleu >= optim.best_metric:
                    saveModel(valid_bleu)
                optim.updateLearningRate(valid_bleu, epoch)

            if testData is not None and totalBatchCount % opt.eval_per_batch == 0 and totalBatchCount >= opt.start_eval_batch:
                model.eval()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'test'))
                test_bleu = evalModel(model, translator, testData)
                logger.info('Test Score (bleu): %g' % (test_bleu * 100))

        print('epoch generation loss: {}'.format(epoch_loss))
        print('epoch mention prediction loss: {}'.format(epoch_cl_loss))
        print('epoch nlu module loss: {}'.format(epoch_nlu_loss))

        return total_loss / total_words, total_num_correct / total_words

    for epoch in range(opt.start_epoch, opt.epochs + 1):

        logger.info('Epoch {}'.format(epoch))
        # train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch)
        # calculate metric for each epoch training
        train_ppl = math.exp(min(train_loss, 100))
        logger.info('Train perplexity: %g' % train_ppl)
        logger.info('Train accuracy: %g' % (train_acc * 100))
        logger.info('Saving checkpoint for epoch {0}...'.format(epoch))
        saveModel()


def main():

    import preprocess
    preprocess.lower = opt.lower_input
    preprocess.seq_length = opt.max_sent_length
    preprocess.shuffle = 1 if opt.process_shuffle else 0
    from preprocess import prepare_data_online

    # opt.train_src (source file of sequence) 'it is a replica of the grotto at lourdes , france where the virgin mary reputedly appeared to saint bernadette soubirous in 1858 .'
    # opt.src_vocab (source file of vocab) 'the(word) 4(index) 256272(frequency) 0.06749202214022335'
    # opt.train_tgt (source file of question) 'to whom did the virgin mary allegedly appear in 1858 in lourdes france ?'
    # opt.tgt_vocab (source file of vocab) same file with opt.src_vocab !!

    dataset = prepare_data_online(opt.train_src, opt.src_vocab, opt.train_tgt, opt.tgt_vocab, opt.train_guide_src, opt.guide_src_vocab)

    trainData = seq2seq.Dataset(dataset['train']['src'], dataset['train']['tgt'], dataset['train']['switch'],
                            dataset['train']['c_tgt'], opt.batch_size, opt.gpus, dataset['train']['guide_src'])

    dicts = dataset['dicts']
    logger.info(' * vocabulary size. source = %d; target = %d' % (dicts['src'].size(), dicts['tgt'].size()))
    logger.info(' * number of training sentences. %d' % len(dataset['train']['src']))
    logger.info(' * maximum batch size. %d' % opt.batch_size)

    logger.info('Building Model ...')
    encoder = seq2seq.Models.Encoder(opt, dicts['src'], dicts['guide_src'])
    decoder = seq2seq.Models.Decoder(opt, dicts['tgt'])
    decIniter = seq2seq.Models.DecInit(opt)

    # generator map output embedding to vocab size vector then softmax
    generator = nn.Sequential(
        nn.Linear(opt.dec_rnn_size // opt.maxout_pool_size, dicts['tgt'].size()),
        nn.Softmax(dim=1)
    )
    classifier = nn.Sequential(
        nn.Linear(opt.dec_rnn_size + 300, dicts['guide_src'].size()),
        nn.Softmax(dim=1)
    )
    nlu_generator = nn.Sequential(
        nn.Linear(opt.dec_rnn_size * 2, dicts['guide_src'].size()),
        nn.Softmax(dim=1)
    )

    model = seq2seq.Models.Seq2Seq_Model(encoder, decoder, decIniter)
    model.generator = generator
    model.classifier = classifier
    model.nlu_generator = nlu_generator
    translator = seq2seq.Translator(opt, model, dataset)

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
        classifier.cuda()
        nlu_generator.cuda()
    else:
        model.cpu()
        generator.cpu()
        classifier.cpu()
        nlu_generator.cpu()

    # trainin with multiple GPUs
    '''
    # if len(opt.gpus) > 1:
    #     model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
    #     generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)
    '''

    for pr_name, p in model.named_parameters():
        logger.info(pr_name)
        # p.data.uniform_(-opt.param_init, opt.param_init)
        if p.dim() == 1:
            # p.data.zero_()
            p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
        else:
            nn.init.xavier_normal_(p, math.sqrt(3))

    encoder.load_pretrained_vectors(opt)
    decoder.load_pretrained_vectors(opt)

    optim = seq2seq.Optim(
        opt.optim, opt.learning_rate,
        max_grad_norm=opt.max_grad_norm,
        max_weight_value=opt.max_weight_value,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at,
        decay_bad_count=opt.halve_lr_bad_count
    )
    optim.set_parameters(model.parameters())

    validData = None
    if opt.dev_input_src and opt.dev_ref:
        validData = load_dev_data(translator, opt.dev_input_src, opt.dev_ref, opt.dev_guide_src)

    testData = None
    if opt.test_input_src and opt.test_ref:
        testData = load_dev_data(translator, opt.test_input_src, opt.test_ref, opt.test_guide_src)

    trainModel(model, translator, trainData, validData, testData, dataset, optim)


if __name__ == "__main__":
    main()

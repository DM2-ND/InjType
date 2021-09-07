from __future__ import division

import os
import xargs
import json
import S2S
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

from bleu_rouge import eval_bleu_rouge

from S2S.xinit import xavier_normal, xavier_uniform
torch.backends.cudnn.enabled=False

parser = argparse.ArgumentParser(description='train.py')
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
def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[S2S.Constants.PAD] = 0

    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def reverse_NMTCriterion(vocabSize):
    weight = torch.zeros(vocabSize)
    weight[S2S.Constants.PAD] = 0

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


def generate_copy_loss_function(g_outputs, c_outputs, g_targets, c_switch, c_targets, c_gate_values,
                                generator, nlu_generator, classifier, crit, copyCrit, reverse_crit, 
                                guideCrit, nluCrit, guides, g_hiddens, nlu_outputs):

    # g_out is output [(TargetLength, BatchSize, Hidden)]
    # c_out is attn_out [(TargetLength, BatchSize, InputLength)]
    # c_gate is copy_prob [(TargetLength, BatchSize, 1)]
    batch_size = g_outputs.size(1)
    targetLength = g_outputs.size(0)
    
    guides = guides.transpose(1,0)
    m_targets = g_targets.clone().detach()
    m_targets = m_targets.transpose(1,0)
    for i in range(batch_size):
        count = 0
        for j in range(targetLength):
            if m_targets[i][j] == S2S.Constants.SS:
                m_targets[i][j] = guides[i][count]
                count += 1
            else:
                m_targets[i][j] = S2S.Constants.PAD

    m_targets = m_targets.transpose(1,0)
    
    nlu_out_t = nlu_outputs.view(-1, nlu_outputs.size(2))
    nlu_prob_t = nlu_generator(nlu_out_t)
    nlu_prob_t = nlu_prob_t.view(-1, batch_size, nlu_prob_t.size(1))

    g_out_t = g_outputs.view(-1, g_outputs.size(2))
    g_prob_t = generator(g_out_t)
    g_prob_t = g_prob_t.view(-1, batch_size, g_prob_t.size(1))
    g_hiddens = g_hiddens.squeeze(1).transpose(1,0)
    
    cl_loss = 0
    g_targets = g_targets.transpose(1,0)
    for i in range(batch_size):
        guide_prob = []
        guide = []
        count = 0
        for j in range(targetLength):
            if g_targets[i][j] == S2S.Constants.SS:
                guide_prob.append(torch.log(classifier(g_hiddens[i][j].unsqueeze(0))))
                guide.append(guides[i][count])
                count += 1
        guide_prob = torch.stack(guide_prob).squeeze(1)
        guide = torch.stack(guide).view(-1)
        if i == 0:
            cl_loss = guideCrit(guide_prob, guide)
        else:
            cl_loss += guideCrit(guide_prob, guide)
    g_targets = g_targets.transpose(1, 0)

    c_output_prob = c_outputs * c_gate_values.expand_as(c_outputs) + 1e-8
    g_output_prob = g_prob_t * (1 - c_gate_values).expand_as(g_prob_t) + 1e-8

    c_output_prob_log = torch.log(c_output_prob)
    g_output_prob_log = torch.log(g_output_prob)
    nlu_prob_log = torch.log(nlu_prob_t)
    c_output_prob_log = c_output_prob_log * (c_switch.unsqueeze(2).expand_as(c_output_prob_log))
    torch.cuda.empty_cache()
    g_output_prob_log = g_output_prob_log * ((1 - c_switch).unsqueeze(2).expand_as(g_output_prob_log))

    g_output_prob_log = g_output_prob_log.view(-1, g_output_prob_log.size(2))
    c_output_prob_log = c_output_prob_log.view(-1, c_output_prob_log.size(2))
    nlu_prob_log = nlu_prob_log.view(-1, nlu_prob_log.size(2))

    g_loss = crit(g_output_prob_log, g_targets.view(-1))
    reverse_g_loss = reverse_crit(g_output_prob_log, g_targets.view(-1))
    c_loss = copyCrit(c_output_prob_log, c_targets.view(-1))
    nlu_loss = nluCrit(nlu_prob_log, m_targets.view(-1))

    total_loss = g_loss + c_loss + nlu_loss * opt.nlu + cl_loss * opt.cls
    report_loss = total_loss.item()
    
    return total_loss, report_loss, g_loss, cl_loss, nlu_loss


def addPair(f1, f2, f3):
    for x, y1, z in zip(f1, f2, f3):
        yield (x, y1, z)
    yield (None, None, None)


def load_dev_test_data(translator, src_file, tgt_file, guide_file):
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
            if len(src_batch) < opt.batch_size: continue
        else:
            if len(src_batch) == 0: break
        
        data = translator.buildData(src_batch, tgt_batch, guide_batch)
        dataset.append(data)
        raw.append((src_batch, tgt_batch))
        src_batch, tgt_batch = [], []
        guide_batch = []
    srcF.close()
    tgtF.close()
    guideF.close()

    return (dataset, raw)


totalBatchCount = 0
def evalModel(model, translator, evalData, mode):
    
    global totalBatchCount
    ofn_prefix = opt.ofn_prefix + f'_{mode}'
    if opt.save_path:
        ofn = os.path.join(opt.save_path, ofn_prefix + f'_{totalBatchCount}.txt')
        ofm = os.path.join(opt.save_path, ofn_prefix + f'.json')

    predict = []
    processed_data, raw_data = evalData
    for batch, raw_batch in zip(processed_data, raw_data):
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
            predBatch.append(
                translator.buildTargetTokens(pred[b][0], src_batch[b], predIsCopy[b][0], predCopyPosition[b][0], attn[b][0]))
        predict += predBatch

    with open(ofn, 'w', encoding='utf-8') as of:
        for pred in predict:
            of.write(' '.join(pred) + '\n')
    
    if mode == 'dev':
        tfn = opt.dev_guide_src
        rfn = opt.dev_ref
    elif mode == 'test':
        tfn = opt.test_guide_src
        rfn = opt.test_ref

    metrics = {'#batch': totalBatchCount}
    metrics.update(eval_bleu_rouge(filepath=ofn, typepath=tfn, refpath=rfn))
    with open(ofm, 'a', encoding='utf-8') as of:
        metrics = json.dumps(metrics)
        of.write(metrics + '\n')


def trainModel(model, translator, trainData, validData, testData, dataset, optim):
    logger.info(model)
    model.train()
    torch.cuda.empty_cache()
    for name,para in model.named_parameters():
        if para.requires_grad:
            print(name,para.data.shape)
    logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))

    # define criterion of each GPU
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())
    guideCriterion = NMTCriterion(dataset['dicts']['guide_src'].size())
    nluCriterion = NMTCriterion(dataset['dicts']['guide_src'].size())
    reverse_criterion = reverse_NMTCriterion(dataset['dicts']['tgt'].size())
    copyLossF = nn.NLLLoss(size_average=False)

    start_time = time.time()

    def saveModel(metric=None):
        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(
            opt.gpus) > 1 else model.generator.state_dict()
        #  (4) drop a checkpoint
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

    def trainEpoch(epoch):
        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words = 0, 0
        report_loss, report_tgt_words, report_src_words = 0, 0, 0
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
            g_outputs, c_outputs, c_gate_values, g_hiddens, nlu_outputs = model(batch, False)
            targets = batch[1][0][1:]  # exclude <s> from targets
            copy_switch = batch[1][1][1:]
            c_targets = batch[1][2][1:]
            g_guides = batch[2][0]

            torch.cuda.empty_cache()
            loss, res_loss, g_loss, cl_loss, nlu_loss = generate_copy_loss_function(
                g_outputs, c_outputs, targets, copy_switch, c_targets, c_gate_values, model.generator, model.nlu_generator, model.classifier,
                criterion, copyLossF, reverse_criterion, guideCriterion, nluCriterion, g_guides, g_hiddens, nlu_outputs)
            torch.cuda.empty_cache()
            epoch_loss += g_loss
            epoch_cl_loss += cl_loss
            epoch_nlu_loss += nlu_loss

            # update the parameters
            torch.cuda.empty_cache()
            loss.backward()
            optim.step()

            num_words = targets.data.ne(S2S.Constants.PAD).sum().item()
            report_loss += res_loss
            report_tgt_words += num_words
            report_src_words += batch[0][-1].data.sum()
            total_loss += res_loss
            total_words += num_words
            if i % opt.log_interval == 0:
                logger.info(
                    "Epoch %2d, %5d/%5d; loss: %6.2f; ppl: %6.2f; %3.0f src token/s; %3.0f tgt token/s" %
                    (epoch, i+1, len(trainData), report_loss,
                     math.exp(min((report_loss / report_tgt_words), 16)),
                     report_src_words / max((time.time() - start), 1.0),
                     report_tgt_words / max((time.time() - start), 1.0)))

                report_loss = report_tgt_words = report_src_words = 0
                start = time.time()
                
            if validData is not None and totalBatchCount % opt.eval_per_batch == 0 and totalBatchCount >= opt.start_eval_batch:
                model.eval()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
                evalModel(model, translator, validData, 'dev')

            if testData is not None and totalBatchCount % opt.eval_per_batch == 0 and totalBatchCount >= opt.start_eval_batch:
                model.eval()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'test'))
                evalModel(model, translator, testData, 'test')

        return total_loss / total_words

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        train_loss = trainEpoch(epoch)
        torch.cuda.empty_cache()
        if opt.save_model:
            saveModel()


def main():

    import dataloader
    dataloader.lower = opt.lower_input
    dataloader.seq_length = opt.max_sent_length
    dataloader.shuffle = 1 if opt.process_shuffle else 0
    from dataloader import prepare_data_online
    dataset = prepare_data_online(opt.train_src, opt.src_vocab, opt.train_tgt, opt.tgt_vocab, opt.train_guide_src, opt.guide_src_vocab)

    trainData = S2S.Dataset(dataset['train']['src'], dataset['train']['tgt'], dataset['train']['switch'], 
        dataset['train']['c_tgt'], opt.batch_size, opt.gpus, dataset['train']['guide_src'])

    dicts = dataset['dicts']
    logger.info(' * vocabulary size. source = %d; target = %d' % (dicts['src'].size(), dicts['tgt'].size()))
    logger.info(' * number of training sentences. %d' % len(dataset['train']['src']))
    logger.info(' * maximum batch size. %d' % opt.batch_size)

    logger.info('Building Model ...')
    encoder = S2S.Models.Encoder(opt, dicts['src'], dicts['guide_src'])
    decoder = S2S.Models.Decoder(opt, dicts['tgt'])
    decIniter = S2S.Models.DecInit(opt)

    ''' generator map output embedding to vocab size vector then softmax'''
    generator = nn.Sequential(
        nn.Linear(opt.dec_rnn_size // opt.maxout_pool_size, dicts['tgt'].size()),
        nn.Softmax(dim=1)
    )
    classifier = nn.Sequential(
        nn.Linear(opt.dec_rnn_size, dicts['guide_src'].size()),
        nn.Softmax(dim=1)
    )
    nlu_generator = nn.Sequential(
        nn.Linear(opt.dec_rnn_size * 2, dicts['guide_src'].size()),
        nn.Softmax(dim=1)
    )

    model = S2S.Models.NMTModel(encoder, decoder, decIniter)
    model.generator = generator
    model.classifier = classifier
    model.nlu_generator = nlu_generator
    translator = S2S.Translator(opt, model, dataset)

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

    for pr_name, p in model.named_parameters():
        logger.info(pr_name)
        if p.dim() == 1:
            p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
        else:
            nn.init.xavier_normal_(p, math.sqrt(3))

    encoder.load_pretrained_vectors(opt)
    decoder.load_pretrained_vectors(opt)

    optim = S2S.Optim(
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
        validData = load_dev_test_data(translator, opt.dev_input_src, opt.dev_ref, opt.dev_guide_src)

    testData = None
    if opt.test_input_src and opt.test_ref:
        testData = load_dev_test_data(translator, opt.test_input_src, opt.test_ref, opt.test_guide_src)

    trainModel(model, translator, trainData, validData, testData, dataset, optim)


if __name__ == "__main__":
    main()

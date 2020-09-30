from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import seq2seq


class Dataset(object):
    def __init__(self, srcData, tgtData, copySwitchData, copyTgtData,
                 batchSize, cuda, guideData):
        self.src = srcData
        self.guide = guideData

        if tgtData:
            self.tgt = tgtData
            # copy switch should company tgt label
            self.copySwitch = copySwitchData
            self.copyTgt = copyTgtData
            assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None
            self.copySwitch = None
            self.copyTgt = None
        self.device = torch.device("cuda" if cuda else "cpu")

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src) / batchSize)

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(seq2seq.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)
        guideBatch, guidelengths = self._batchify(
            self.guide[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index * self.batchSize:(index + 1) * self.batchSize])

        else:
            tgtBatch = None

        if self.copySwitch is not None:
            copySwitchBatch = self._batchify(
                self.copySwitch[index * self.batchSize:(index + 1) * self.batchSize])
            copyTgtBatch = self._batchify(
                self.copyTgt[index * self.batchSize:(index + 1) * self.batchSize])
        else:
            copySwitchBatch = None
            copyTgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        if tgtBatch is None:
            batch = zip(indices, srcBatch, guideBatch)
        else:
            if self.copySwitch is not None:
                batch = zip(indices, srcBatch, tgtBatch, copySwitchBatch, copyTgtBatch, guideBatch)
            else:
                batch = zip(indices, srcBatch, tgtBatch, guideBatch)
        # batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch, guideBatch = zip(*batch)
        else:
            if self.copySwitch is not None:
                indices, srcBatch, tgtBatch, copySwitchBatch, copyTgtBatch, guideBatch = zip(*batch)
            else:
                indices, srcBatch, tgtBatch, guideBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            b = b.to(self.device)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)

        return (wrap(srcBatch), lengths), \
               (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
               (wrap(guideBatch), lengths), \
               indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(
            zip(self.src, self.tgt, self.copySwitch, self.copyTgt, self.guide))
        self.src, self.tgt, self.copySwitch, self.copyTgt, self.guide = zip(
            *[data[i] for i in torch.randperm(len(data))])

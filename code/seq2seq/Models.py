import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import seq2seq.modules
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

try:
    import ipdb
except ImportError:
    pass


class Encoder(nn.Module):
    def __init__(self, opt, dicts, guide_dicts):

        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.enc_rnn_size % self.num_directions == 0
        self.hidden_size = opt.enc_rnn_size // self.num_directions
        self.input_size = opt.word_vec_size

        super(Encoder, self).__init__()

        self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=seq2seq.Constants.PAD)
        self.guide_lut = nn.Embedding(guide_dicts.size(), opt.word_vec_size, padding_idx=seq2seq.Constants.PAD)

        self.word_gate = nn.Parameter(torch.tensor([1.]))
        self.guide_gate = nn.Parameter(torch.tensor([1.]))

        self.input_size = self.input_size * 2

        self.rnn = nn.GRU(self.input_size, self.hidden_size, num_layers=opt.layers, dropout=opt.dropout, bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, guide, hidden=None):

        # input: (wrap(srcBatch), wrap(srcBioBatch), lengths)
        # lengths data is wrapped inside a Variable
        input_lengths = input[-1].data.view(-1).tolist()

        # wordEmb is the embedding of type level input
        wordEmb = self.word_lut(input[0])
        # print(wordEmb.shape)

        # guideEmb is the embedding of mention level input
        guideEmb = self.guide_gate * self.guide_lut(guide[0])

        input_emb = torch.cat((wordEmb, guideEmb), dim=-1)

        emb = pack_padded_sequence(input_emb, input_lengths)

        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = pad_packed_sequence(outputs)[0]

        # hidden_t == torch.Size([2, 32, 256]) == (bi-GRU, batch, hidden_dim)
        # outputs == torch.Size([17, 32, 512]) == (seq_len, batch, hidden_dim * 2)
        # wordEmb == torch.Size([17, 32, 300]) == (seq_len, batch, hidden_dim)

        return hidden_t, outputs, wordEmb


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):

        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            if hidden == None:
                h_1_i = layer(input, None)
            else:
                h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class Decoder(nn.Module):
    def __init__(self, opt, dicts):

        self.opt = opt
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.enc_rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=seq2seq.Constants.PAD)

        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        self.reverse_rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)

        self.attn = seq2seq.modules.ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size, opt.att_vec_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.readout = nn.Linear((opt.enc_rnn_size + opt.dec_rnn_size + opt.word_vec_size), opt.dec_rnn_size)

        self.maxout = seq2seq.modules.MaxOut(opt.maxout_pool_size)
        self.maxout_pool_size = opt.maxout_pool_size
        self.nextEnt_gate = nn.Parameter(torch.tensor([1.]))

        self.copySwitch = nn.Linear(opt.enc_rnn_size + opt.dec_rnn_size, 1)

        self.hidden_size = opt.dec_rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, src, input, hidden, context, src_pad_mask, init_att, evaluate, wordEmb, nlu):
        # input vector (22(TargetLength!), BatchSize)
        # hidden vector (1, BatchSize=64, 2(Bi)*HiddenSize)
        # context is output of encoder (SeqLength, BatchSize, 2(Bi)*HiddenSize)
        # src_pad_mask indicates which item is padded (BatchSize, SeqLength) e.g, (64, 61)

        wordEmb = self.word_lut(input)

        # while training, use teacher forcing and use input type for prediction
        if not evaluate:
            src = src[0].transpose(1, 0) # transform into (BatchSize, ...) -> batch first!
            mod_input = input.clone().detach()
            mod_input = mod_input.transpose(1, 0) # input vector (BatchSize, 22(TargetLength!))
            src_ind = [0 for i in range(src.size()[0])]
            for i in range(mod_input.size()[0]): # for batch
                for j in range(mod_input.size()[1]): # for sentence length
                    if mod_input[i][j] == seq2seq.Constants.SS:
                        mod_input[i][j] = src[i][src_ind[i]]
                        src_ind[i] += 1
            mod_input = mod_input.transpose(1, 0)
            wordEmb = self.word_lut(mod_input)

        # when nlu is used for multi-tasking, add reverse embedding of the target sentence
        if nlu:
            wordEmb = self.word_lut(mod_input)
            reverse_input = input.clone().detach()
            reverse_input = reverse_input.transpose(1,0)
            reverse_input = reverse_input.cpu().numpy()
            for i in range(len(reverse_input)):
                for j in range(len(reverse_input[0])):
                    if reverse_input[i][j] == 0:
                        reverse_input[i][:j] = reverse_input[i][:j][::-1]
                        break

            reverse_input = torch.cuda.LongTensor(reverse_input)
            reverse_input = reverse_input.transpose(1,0)
            reverse_emb = self.word_lut(reverse_input)

        g_outputs = []
        c_outputs = []
        g_hiddens = []
        reverse_g_outputs = []
        copyGateOutputs = []
        cur_context = init_att
        self.attn.applyMask(src_pad_mask)
        precompute = None

        for emb_t in wordEmb.split(1):

            # wordEmb.split(1) generate each target word embedding ->
            # For example [seqlength(22), 64, 300)] -> 22 * [(1, 64, 300)]

            # emb_t.shape [(1, batch, decoder_hidden_size)]
            emb_t = emb_t.squeeze(0) # [(1, batch, decoder_hidden_size)] -> [(batch, decoder_hidden_size)]

            input_emb = emb_t
            if self.input_feed:
                input_emb = torch.cat([emb_t, cur_context], dim=1)

            # emb_t = [(64, 512(context) + 300(input))] and hidden [(64, 512)]
            output, hidden = self.rnn(input_emb, hidden)
            # when not using nlu, apply attention on input list, otherwise, do not apply attention

            if not nlu:
                cur_context, attn, precompute = self.attn(output, context.transpose(0, 1), precompute)

                # cur_context is from attn [(64, 512)] and attn [(64, Seqlength)]
                # precompute is encoder output [(64, Seqlength, 512)]
                copyProb = self.copySwitch(torch.cat((output, cur_context), dim=1)) # copyProb = [(64, 1)]

                # when calculate copyProb, we need cur_context (attn)
                copyProb = F.sigmoid(copyProb) # In (0, 1) for copy probability -> pointer generator network

                # readout -> (64, 512) then maxout -> (64, 256)
                readout = self.readout(torch.cat((emb_t, output, cur_context), dim=1))
                maxout = self.maxout(readout)
                output = self.dropout(maxout)

                g_hiddens += [hidden]
                g_outputs += [output]
                c_outputs += [attn]
                copyGateOutputs += [copyProb]

            else:
                g_outputs += [output]

        # when using nlu, combine forward prediction from shared decoder and the reverse prediction from the right-to-left language model
        if nlu:
            for emb_t in reverse_emb.split(1):
                emb_t = emb_t.squeeze(0)

                # emb_t.shape [(64, 300)]
                input_emb = emb_t
                input_emb = torch.cat([emb_t, cur_context], 1)
                output, hidden = self.reverse_rnn(input_emb, hidden)

                reverse_g_outputs += [output]
            g_outputs = torch.stack(g_outputs)
            reverse_g_outputs = torch.stack(reverse_g_outputs)
            nlu_outputs = torch.cat([g_outputs, reverse_g_outputs],2)

            return nlu_outputs

        g_outputs = torch.stack(g_outputs)
        c_outputs = torch.stack(c_outputs)
        g_hiddens = torch.stack(g_hiddens)
        copyGateOutputs = torch.stack(copyGateOutputs)

        # g_output is readout vector -> use for project to vocab
        # c_output is attention vectior
        # copyGate is copy probability range (0, 1)

        return g_outputs, c_outputs, copyGateOutputs, hidden, attn, cur_context, g_hiddens


''' Decoder Initation: Last_Hidden_Size to Dec_Rnn_Size'''
class DecInit(nn.Module):
    def __init__(self, opt):
        super(DecInit, self).__init__()

        self.num_directions = 2 if opt.brnn else 1
        assert opt.enc_rnn_size % self.num_directions == 0
        self.enc_rnn_size = opt.enc_rnn_size
        self.dec_rnn_size = opt.dec_rnn_size

        self.tanh = nn.Tanh()
        self.initer = nn.Linear(self.enc_rnn_size // self.num_directions, self.dec_rnn_size)

    ''' [BatchSize, Hidden_Size] -> [BatchSize, Dec_Size]  '''
    ''' [BatchSize, Hidden_Size] -> (64, 256) to (64, 512) '''
    def forward(self, last_enc_h):
        return self.tanh(self.initer(last_enc_h))


class Seq2Seq_Model(nn.Module):
    def __init__(self, encoder, decoder, decIniter):
        super(Seq2Seq_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decIniter = decIniter

    def make_init_att(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.encoder.hidden_size * self.encoder.num_directions)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def forward(self, input, evaluate):

        '''
            input stucture:
            [0] (wrap(srcBatch), lengths),
            [1] (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)),
            [2] (wrap(guideBatch), lengths),
        '''

        type_input, mention_input = input[0], input[2]
        target_output = input[1][0][:-1]  # exclude last target from inputs
        input_pad_mask = Variable(type_input[0].data.eq(seq2seq.Constants.PAD).transpose(0, 1).float(), requires_grad=False, volatile=False)

        enc_hidden, context, wordEmb = self.encoder(type_input, mention_input)

        init_att = self.make_init_att(context)
        enc_hidden = self.decIniter(enc_hidden[1]).unsqueeze(0)  # [1] is the last backward hiden

        g_out, c_out, c_gate_out, dec_hidden, _attn, _attention_vector, g_hid = self.decoder(type_input, target_output, enc_hidden, context, input_pad_mask, init_att, evaluate, wordEmb, False)

        nlu_target_output = input[1][0][1:] # exclude fast target from inputs
        nlu_output = self.decoder(type_input, nlu_target_output, None, context, input_pad_mask, init_att, evaluate, wordEmb, True)

        # g_out is output [(TargetLength, BatchSize, Hidden)]
        # c_out is attn_out [(TargetLength, BatchSize, InputLength)]
        # c_gate is copy_prob [(TargetLength, BatchSize, 1)], which is part of legacy code and is not used in this project
        # nlu_output is nlu predicted output [(TargetLength, BatchSize, Hidden)]

        if evaluate:
            return g_out, c_out, c_gate_out, g_hid
        else:
            return g_out, c_out, c_gate_out, g_hid, nlu_output, wordEmb

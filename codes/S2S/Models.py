import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import S2S.modules
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
        self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=S2S.Constants.PAD)
        self.guide_lut = nn.Embedding(guide_dicts.size(), opt.word_vec_size, padding_idx=S2S.Constants.PAD)
        self.word_gate = nn.Parameter(torch.tensor([1.]))
        self.guide_gate = nn.Parameter(torch.tensor([1.]))
        self.input_size = self.input_size * 2

        self.rnn = nn.GRU(self.input_size, self.hidden_size, num_layers=opt.layers, dropout=opt.dropout, bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, guide, hidden=None):

        # lengths data is wrapped inside a Variable
        lengths = input[-1].data.view(-1).tolist()
        
        # load word embedding
        # torch.Size([SeqLength, BatchSize=64, WordDim=300])
        # seqlength for each batch is different
        wordEmb = self.word_lut(input[0])
        guideEmb = self.guide_gate * self.guide_lut(guide[0])

        input_emb = torch.cat((wordEmb, guideEmb), dim=-1)
        
        backward_hids = []
        for i in range(input_emb.size(0)):
            output, hidden = self.rnn(input_emb[i].unsqueeze(0), hidden)
            backward_hids.append(hidden[1])
        backward_hids = torch.stack(backward_hids)

        hidden = None

        emb = pack_padded_sequence(input_emb, lengths)
            
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = pad_packed_sequence(outputs)[0]

        return hidden_t, outputs, backward_hids, wordEmb


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
        self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=S2S.Constants.PAD)
        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        self.reverse_rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        self.attn = S2S.modules.ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size, opt.att_vec_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.readout = nn.Linear((opt.enc_rnn_size + opt.dec_rnn_size + opt.word_vec_size), opt.dec_rnn_size)
        self.maxout = S2S.modules.MaxOut(opt.maxout_pool_size)
        self.maxout_pool_size = opt.maxout_pool_size
        self.nextEnt_gate = nn.Parameter(torch.tensor([1.]))

        self.copySwitch = nn.Linear(opt.enc_rnn_size + opt.dec_rnn_size, 1)

        self.hidden_size = opt.dec_rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, src, input, hidden, context, src_pad_mask, init_att, evaluate, nlu, backward_hids, wordEmb, nextCounts):
        # input vector (22(TargetLength!), BatchSize)
        # hidden vector (1, BatchSize=64, 2(Bi)*HiddenSize)
        # context is output of encoder (SeqLength, BatchSize, 2(Bi)*HiddenSize)
        # src_pad_mask indicates which item is padded (BatchSize, SeqLength) e.g, (64, 61)
        
        emb = self.word_lut(input)
        if evaluate:
            batch_size = input.size(1)
            input = input.transpose(1,0)
            nextEnt_embs = torch.zeros([input.size(0),input.size(1),emb.size(2)],device='cuda:0',requires_grad=False)
            wordEmb = wordEmb.transpose(1,0)
            for i in range(input.size(0)): # for batch
                try:
                    nextEnt_embs[i][0] = wordEmb[i][nextCounts[i]]
                except:
                    pass
            nextEnt_embs = nextEnt_embs.transpose(1,0)
            input = input.transpose(1,0)
            #emb = emb + self.nextEnt_gate * nextEnt_embs

        else:
            # prepare the emb of next entity word
            batch_size = input.size(1)
            next_counts = [0 for i in range(batch_size)]
            input = input.transpose(1,0)
            nextEnt_embs = torch.zeros([input.size(0),input.size(1),emb.size(2)],device='cuda:0',requires_grad=False)
            wordEmb = wordEmb.transpose(1,0)
            for i in range(input.size(0)): # for batch
                for j in range(input.size(1)): # for sentence length
                    try:
                        nextEnt_embs[i][j] = wordEmb[i][next_counts[i]]
                    except:
                        pass
                    if input[i][j] == S2S.Constants.SS:
                        next_counts[i] += 1
            nextEnt_embs = nextEnt_embs.transpose(1,0)
            input = input.transpose(1,0)

        if not evaluate:
            src = src[0].transpose(1,0) # transform into (BatchSize, 10(SrcLength))
            mod_input = input.clone().detach()
            mod_input = mod_input.transpose(1,0) # input vector (BatchSize, 22(TargetLength!))
            src_ind = [0 for i in range(src.size()[0])]
            for i in range(mod_input.size()[0]): # for batch
                for j in range(mod_input.size()[1]): # for sentence length
                    try:
                        if mod_input[i][j] == S2S.Constants.SS:
                            mod_input[i][j] = src[i][src_ind[i]]
                            src_ind[i] += 1
                    except:
                        print(mod_input[i])
            mod_input = mod_input.transpose(1,0)

            emb = self.word_lut(mod_input)
        
        if nlu:
            emb = self.word_lut(mod_input)
            reverse_input = mod_input.clone().detach()
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
        
        for emb_t in emb.split(1):
            # emb_t.split(1) generate each target word embedding
            # For example [(22(seqlength), 64, 300)] -> 22 * [(1, 64, 300)]
            # emb_t.shape [(1, 64, 300)]
            emb_t = emb_t.squeeze(0)

            # emb_t.shape [(64, 300)]
            input_emb = emb_t
            if self.input_feed:
                input_emb = torch.cat([emb_t, cur_context], 1)

            # input_emb = [(64, 512(context) + 300(input))] and hidden [(64, 512)]
            output, hidden = self.rnn(input_emb, hidden)
            if not nlu:
                cur_context, attn, precompute = self.attn(output, context.transpose(0, 1), precompute)

                # cur_context is from attn [(64, 512)] and attn [(64, Seqlength)]
                # precompute is encoder output [(64, Seqlength, 512)]
                copyProb = self.copySwitch(torch.cat((output, cur_context), dim=1))
                # copyProb = [(64, 1)]
                # when calculate copyProb, we need cur_context (attn)
                copyProb = F.sigmoid(copyProb)

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

    ''' [BatchSize, Hidden_Size] -> [BatchSize, Dec_Size] (64, 256) -> (64, 512)'''
    def forward(self, last_enc_h):
        # batchSize = last_enc_h.size(0)
        # dim = last_enc_h.size(1)
        return self.tanh(self.initer(last_enc_h))


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, decIniter):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decIniter = decIniter

    def make_init_att(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.encoder.hidden_size * self.encoder.num_directions)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def forward(self, input, evaluate):

        # ipdb.set_trace()
        src = input[0]
        tgt = input[1][0][:-1]  # exclude last target from inputs
        src_pad_mask = Variable(src[0].data.eq(S2S.Constants.PAD).transpose(0, 1).float(), requires_grad=False, volatile=False)
        guide = input[2]

        enc_hidden, context, backward_hids, wordEmb = self.encoder(src, guide)

        init_att = self.make_init_att(context)
        enc_hidden = self.decIniter(enc_hidden[1]).unsqueeze(0)  # [1] is the last backward hiden

        g_out, c_out, c_gate_out, dec_hidden, _attn, _attention_vector, g_hid = self.decoder(src, tgt, enc_hidden, context, src_pad_mask, init_att, evaluate, False, backward_hids, wordEmb, [])

        tgt = input[1][0][1:]
        nlu_output = self.decoder(src, tgt, None, context, src_pad_mask, init_att, evaluate, True, backward_hids, wordEmb, [])

        if evaluate:
            return g_out, c_out, c_gate_out, g_hid
        else:
            return g_out, c_out, c_gate_out, g_hid, nlu_output

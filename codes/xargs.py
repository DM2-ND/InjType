import argparse

def add_data_options(parser):
    ## Data options
    parser.add_argument('-save_model', default=True,
                        help="""Whether to save model (True of False)""")
    parser.add_argument('-save_path', default='',
                        help="""Model save path""")
    parser.add_argument('-train_from_state_dict', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model.""")
    parser.add_argument('-ofn_prefix', default='', type=str,
                        help="""saved file name prefix""")

    # tmp solution for load issue
    parser.add_argument('-online_process_data', action="store_true")
    parser.add_argument('-process_shuffle', action="store_true")
    parser.add_argument('-train_src', help='Path to the train input file (type level).')
    parser.add_argument('-train_guide_src', help='Path to the train input file (mention level).')
    parser.add_argument('-guide_src_vocab', help='Path to the source vocab (mention level).')
    parser.add_argument('-src_vocab', help='Path to the source vocab (type level).')
    parser.add_argument('-train_tgt', help='Path to the output file (no mentions).')
    parser.add_argument('-train_guide_tgt', help='Path to the output file (mention level).')
    parser.add_argument('-tgt_vocab', help='Path to the target vocab (mention level).')
    parser.add_argument('-lower_input', action="store_true",
                        help="Lower case all the input. Default is False")

    # Test options
    parser.add_argument('-dev_input_src', help='Path to the dev input file (type level).')
    parser.add_argument('-dev_guide_src', help='Path to the dev input file (mention level).')
    parser.add_argument('-dev_ref', help='Path to the dev reference file (no mentions).')
    parser.add_argument('-dev_guide_tgt', help='Path to the dev reference file (mention level).')
    parser.add_argument('-test_input_src', help='Path to the test input file.')
    parser.add_argument('-test_guide_src', help='Path to the dev input file.')
    parser.add_argument('-test_ref', help='Path to the test reference file.')
    parser.add_argument('-test_guide_tgt', help='Path to the test reference file (mention level).')
    parser.add_argument('-beam_size', type=int, default=1, help='Beam size')
    parser.add_argument('-max_sent_length', type=int, default=160, help='Maximum sentence length.')
    parser.add_argument('-cls', type=float, default=0, help='Class weight.')
    parser.add_argument('-nlu', type=float, default=0, help='NLU weight')
    

def add_model_options(parser):
    ## Model options
    parser.add_argument('-layers', type=int, default=1,
                        help='Number of layers in the LSTM encoder/decoder')
    parser.add_argument('-enc_rnn_size', type=int, default=256,
                        help='Size of LSTM hidden states')
    parser.add_argument('-dec_rnn_size', type=int, default=256,
                        help='Size of LSTM hidden states')
    parser.add_argument('-word_vec_size', type=int, default=300,
                        help='Word embedding sizes')
    parser.add_argument('-att_vec_size', type=int, default=256,
                        help='Concat attention vector sizes')
    parser.add_argument('-maxout_pool_size', type=int, default=2,
                        help='Pooling size for MaxOut layer.')
    parser.add_argument('-input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")
    parser.add_argument('-brnn',type=bool, default=True,
                        help='Use a bidirectional encoder')
    parser.add_argument('-brnn_merge', default='concat',
                        help="""Merge action for the bidirectional hidden states:
                        [concat|sum]""")


def add_train_options(parser):
    ## Optimization options
    parser.add_argument('-batch_size', type=int, default=32,
                        help='Maximum batch size')
    parser.add_argument('-max_generator_batches', type=int, default=16,
                        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but uses
                        more memory.""")
    parser.add_argument('-epochs', type=int, default=13,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init)""")
    parser.add_argument('-optim', default='adam',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-max_weight_value', type=float, default=15,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.5,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-curriculum', type=int, default=1,
                        help="""For this many epochs, order the minibatches based
                        on source sequence length. Sometimes setting this to 1 will
                        increase convergence speed.""")
    parser.add_argument('-extra_shuffle', action="store_true",
                        help="""By default only shuffle mini-batch order; when true,
                        shuffle and re-assign mini-batches""")
    parser.add_argument('-level', default='entity',
                        help="""Level of input representation""")
    parser.add_argument('-copy', action='store_true',
                        help="""Use copy or not""")
    parser.add_argument('-guide', action='store_true',
                        help="""Use guide or not""")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=1.0,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-start_eval_batch', type=int, default=1,
                        help="""evaluate on dev starting from x-th batch.""")
    parser.add_argument('-eval_per_batch', type=int, default=2000,
                        help="""evaluate on dev per x batches.""")
    parser.add_argument('-halve_lr_bad_count', type=int, default=3)

    # pretrained word vectors
    parser.add_argument('-pretrain', action="store_true",
                        help="""When true, use pretrained embeddings""")
    parser.add_argument('-pre_word_vecs_enc',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")

    # GPU
    parser.add_argument('-gpus', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-log_interval', type=int, default=100,
                        help="logger.info stats at this interval.")
    parser.add_argument('-seed', type=int, default=-1,
                        help="""Random seed used for the experiments
                        reproducibility.""")
    parser.add_argument('-cuda_seed', type=int, default=-1,
                        help="""Random CUDA seed used for the experiments
                        reproducibility.""")
    parser.add_argument('-log_home', default='',
                        help="""log home""")

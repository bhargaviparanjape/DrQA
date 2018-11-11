#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the RNN based DrQA reader."""

import torch
import torch.nn as nn
from . import layers
import pdb


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class RnnSentSelector(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnSentSelector, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Projection for attention weighted question
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        # RNN document encoder
        ## this needs to run across sentences
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        self.self_attn_document = layers.LinearSeqAttn(doc_hidden_size)

        # Bilinear attention for span start/end
        ## simple mlp to score these sentences
        self.sentence_scorer = layers.BilinearSeq(
            doc_hidden_size,
            question_hidden_size,
            doc_hidden_size,
        )

        # self.end_attn = layers.BilinearSeqAttn(
        #     doc_hidden_size,
        #     question_hidden_size,
        #     normalize=normalize,
        # )
        self.relevance_scorer = nn.Sequential(
            nn.Linear(doc_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

        self.relevance_scorer1 = layers.BilinearSeq(
            doc_hidden_size,
            question_hidden_size,
            1,
        )



    def forward(self, x1, x1_f, x1_mask, x1_sent_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Maximum sentences in batch
        max_sent = x1_emb.shape[1]
        batch_size = x1_emb.shape[0]

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Expand Question embeddings to max no. of sentences
        x2_emb_expanded = x2_emb.unsqueeze(1).expand(x2_emb.shape[0], max_sent, x2_emb.shape[1],x2_emb.shape[2]).contiguous()
        x2_mask_expanded = x2_mask.unsqueeze(1).expand(x2_mask.shape[0], max_sent, x2_mask.shape[1]).contiguous()

        # Change views of document vectors and question vectors, add to batch dimension
        x1_emb_flattened = x1_emb.view(x1_emb.shape[0] * max_sent, -1, x1_emb.shape[3])
        x2_emb_flattened = x2_emb_expanded.view(x2_emb_expanded.shape[0] * max_sent, -1, x2_emb_expanded.shape[3])

        # Change views of mask variables
        x1_mask_flattened = x1_mask.view(x1_mask.shape[0] * max_sent, -1)
        x2_mask_flattened = x2_mask_expanded.view(x2_mask_expanded.shape[0] * max_sent, -1)

        # Form document encoding inputs
        drnn_input = [x1_emb_flattened]

        # Add attention-weighted question representation
        # Same question across all sentences of a document
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb_flattened, x2_emb_flattened, x2_mask_flattened)
            drnn_input.append(x2_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            x1_f_flattened =  x1_f.view(x1_f.shape[0] * max_sent, -1, x1_f.shape[3])
            drnn_input.append(x1_f_flattened)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask_flattened)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb_flattened, x2_mask_flattened)

        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask_flattened)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask_flattened)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)


        ### Simple document encoding ###
        # d_merge_weights = layers.uniform_weights(doc_hiddens, x1_mask_flattened)
        d_merge_weights = self.self_attn_document(doc_hiddens, x1_mask_flattened)
        docs_hidden = layers.weighted_avg(doc_hiddens, d_merge_weights)
        relevance_scores = self.relevance_scorer1(docs_hidden, question_hidden).view(batch_size, max_sent, -1).squeeze(2)


        ### Fancy interaction between question and hidden layer ###
        '''
        # Repeat question_hidden for sequence length of document
        question_hidden_expaned = question_hidden.unsqueeze(1).expand(question_hidden.shape[0], doc_hiddens.shape[1], question_hidden.shape[1]).contiguous()
        scores = self.sentence_scorer(doc_hiddens, question_hidden_expaned)
        # Max of the scores (needs to be masked)
        max_scores = \
        scores.data.masked_fill_(x1_mask_flattened.unsqueeze(-1).expand(scores.size()).data, -float("inf")).max(1)[0]
        # Weight vector to predict 2 values
        relevance_scores = self.relevance_scorer(max_scores).view(batch_size, max_sent, -1).squeeze(2)
        '''

        return relevance_scores

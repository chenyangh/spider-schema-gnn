from pytorch_transformers import XLNetConfig, XLNetPreTrainedModel, XLNetTokenizer
from pytorch_transformers.modeling_xlnet import XLNetLayer
import torch
from torch import nn
import numpy as np

MAX_SEQ_LEN = 128
model_name = 'xlnet-large-cased'


class XLNetModel(XLNetPreTrainedModel):

    def __init__(self, config):
        super(XLNetModel, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer

        self.word_embedding = nn.Embedding(config.n_token, config.d_model)
        self.mask_emb = nn.Parameter(torch.Tensor(1, 1, config.d_model))
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)

        self.apply(self.init_weights)

    def _resize_token_embeddings(self, new_num_tokens):
        self.word_embedding = self._get_resized_embeddings(self.word_embedding, new_num_tokens)
        return self.word_embedding

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def create_mask(self, qlen, mlen):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.

        Args:
            qlen: TODO Lysandre didn't fill
            mlen: TODO Lysandre didn't fill

        ::

                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]

        """
        attn_mask = torch.ones([qlen, qlen])
        mask_up = torch.triu(attn_mask, diagonal=1)
        attn_mask_pad = torch.zeros([qlen, mlen])
        ret = torch.cat([attn_mask_pad, mask_up], dim=1)
        if self.same_length:
            mask_lo = torch.tril(attn_mask, diagonal=-1)
            ret = torch.cat([ret[:, :qlen] + mask_lo, ret[:, qlen:]], dim=1)

        ret = ret.to(next(self.parameters()))
        return ret

    def cache_mem(self, curr_out, prev_mem):
        """cache hidden states into memory."""
        if self.mem_len is None or self.mem_len == 0:
            return None
        else:
            if self.reuse_len is not None and self.reuse_len > 0:
                curr_out = curr_out[:self.reuse_len]

            if prev_mem is None:
                new_mem = curr_out[-self.mem_len:]
            else:
                new_mem = torch.cat([prev_mem, curr_out], dim=0)[-self.mem_len:]

        return new_mem.detach()

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum('i,d->id', pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        """create relative positional encoding."""
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))

        if self.attn_type == 'bi':
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
        elif self.attn_type == 'uni':
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            raise ValueError('Unknown `attn_type` {}.'.format(self.attn_type))

        if self.bi_data:
            fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.float)
            bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.float)

            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)

            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz//2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz//2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        pos_emb = pos_emb.to(next(self.parameters()))
        return pos_emb

    def forward(self, input_ids, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, head_mask=None):
        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        input_ids = input_ids.transpose(0, 1).contiguous()
        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None

        qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        mlen = mems[0].shape[0] if mems is not None else 0
        klen = mlen + qlen

        dtype_float = next(self.parameters()).dtype
        device = next(self.parameters()).device

        ##### Attention mask
        # causal attention mask
        if self.attn_type == 'uni':
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError('Unsupported attention type: {}'.format(self.attn_type))

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatbility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
            data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            non_tgt_mask = None

        ##### Word embeddings and prepare h & g hidden states
        word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
        # else:  # We removed the inp_q input which was same as target mapping
        #     inp_q_ext = inp_q[:, :, None]
        #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        ##### Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            mem_pad = torch.zeros([mlen, bsz], dtype=torch.long, device=device)
            cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            # seg_mat = F.one_hot(seg_mat, num_classes=2).to(dtype_float)

            nb_classes = 2
            one_hot_targets = np.eye(nb_classes)[seg_mat.cpu().numpy()]
            seg_mat = torch.from_numpy(one_hot_targets).to(dtype_float).cuda()
        else:
            seg_mat = None

        ##### Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = []
        hidden_states = []
        for i, layer_module in enumerate(self.layer):
            # cache new mems
            new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if self.output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module(output_h, output_g, attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask,
                                   r=pos_emb, seg_mat=seg_mat, mems=mems[i], target_mapping=target_mapping,
                                   head_mask=head_mask[i])
            output_h, output_g = outputs[:2]
            if self.output_attentions:
                attentions.append(outputs[2])

        # Add last hidden state
        if self.output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        outputs = (output.permute(1, 0, 2).contiguous(), new_mems)
        if self.output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(hs.permute(1, 0, 2).contiguous() for hs in hidden_states)
            outputs = outputs + (hidden_states,)
        if self.output_attentions:
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
            outputs = outputs + (attentions,)

        return outputs  # outputs, new_mems, (hidden_states), (attentions)


class XLNetForFeatureExtraction(XLNetPreTrainedModel):
    def __init__(self, config):
        super(XLNetForFeatureExtraction, self).__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None,
                labels=None, head_mask=None):
        transformer_outputs = self.transformer(input_ids, token_type_ids=token_type_ids,
                                               input_mask=input_mask, attention_mask=attention_mask,
                                               mems=mems, perm_mask=perm_mask, target_mapping=target_mapping,
                                               head_mask=head_mask)
        output = transformer_outputs[0]
        return output


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):

    features = []
    for idx, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features


# examples = ["Hello, my dog is cute", "my dog is cute"]
def xlnet_feature_extractor(examples):
    config = XLNetConfig.from_pretrained(model_name)
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    model = XLNetForFeatureExtraction(config)
    # input_ids = torch.tensor(tokenizer.encode(utterance_list[0])).unsqueeze(0)# Batch size 1

    features = convert_examples_to_features(examples, MAX_SEQ_LEN, tokenizer,
                                            cls_token_at_end=True,  # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=2 if True else 1,
                                            pad_on_left=True,  # pad on the left for xlnet
                                            pad_token_segment_id=4 if True else 0)

    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    for feature in features:
        input_ids_list.append(feature.input_ids)
        input_mask_list.append(feature.input_mask)
        segment_ids_list.append(feature.segment_ids)

    input_ids_tensor = torch.tensor(input_ids_list)
    input_mask_tensor = torch.tensor(input_mask_list)
    segment_ids_tensor = torch.tensor(segment_ids_list)

    transformer_outputs = model(input_ids=input_ids_tensor, attention_mask=input_mask_tensor,
                                token_type_ids=segment_ids_tensor)
    feature = transformer_outputs[:, -1]
    return feature
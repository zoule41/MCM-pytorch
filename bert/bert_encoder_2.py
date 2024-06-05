import torch
import torch.nn as nn
from .bert_2 import MyBertMaskedLM
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoTokenizer, BertConfig, BertTokenizer, logging
from .med import BertModel
class BertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [bert-base-uncased](https://huggingface.co/bert-base-uncased) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
    Examples:
    ```python
    >>> from transformers import BertModel, BertConfig
    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> configuration = BertConfig()
    >>> # Initializing a model from the bert-base-uncased style configuration
    >>> model = BertModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bert"

    def __init__(
            self,
            vocab_size=30000,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=100,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()

        self.model = MyBertMaskedLM(BertConfig())

    def forward(self, latent, complaint_feat, ids, labels, attn_mask, token_type_ids):

        outputs = self.model(latent, complaint_feat, ids, attn_mask, token_type_ids, labels=labels)

        return outputs

class BertEncoder2(nn.Module):
    def __init__(self,
                 tokenizer: BertTokenizer = None,
                 emb_dim: int = 768,
                 output_dim: int = 128,
                 hidden_dim: int = 2048,
                 freeze_bert: bool = True):
        super(BertEncoder2, self).__init__()
        self.bert_type = "/root/autodl-tmp/Bio_ClinicalBERT"
        self.last_n_layers = 1
        self.aggregate_method = "sum"
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_bert = True
        self.agg_tokens = True
        # self.max_sent_num = 10

        self.config = BertConfig.from_json_file("/root/autodl-tmp/project/MCM/bert/bert_config.json")
        self.model = BertModel.from_pretrained(
            self.bert_type,
            config=self.config,
            add_pooling_layer=False,
        )

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)

        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
        '''
        :param embeddings: bz, 1, 112, 768
        :param caption_ids: bz, 112
        :param last_layer_attn: bz, 111
        '''
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        last_attns = []

        # loop over batch
        for embs, caption_id, last_attn in zip(embeddings, caption_ids, last_layer_attn):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []
            attns = []
            attn_bank = []

            # loop over sentence
            for word_emb, word_id, attn in zip(embs, caption_id, last_attn):
                word = self.idxtoword[word_id.item()]
                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    attns.append(sum(attn_bank))
                    agg_embs.append(word_emb)
                    words.append(word)
                    attns.append(attn)
                    break
                # This is because some words are divided into two words.
                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(attn)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                        attns.append(sum(attn_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                        attn_bank = [attn]
                else:
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])
                    attn_bank.append(attn)
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.type_as(agg_embs)
            words = words + ["[PAD]"] * padding_size
            last_attns.append(
                torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        last_atten_pt = torch.stack(last_attns)
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)

        return agg_embs_batch, sentences, last_atten_pt

    def forward(self, ids, attn_mask, token_type, get_local=False):
        outputs = self.model(ids, attn_mask, token_type,
                             return_dict=True, mode="text")

        last_layer_attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1)
        all_feat = outputs.last_hidden_state.unsqueeze(1)

        if self.agg_tokens:
            all_feat, sents, last_atten_pt = self.aggregate_tokens(
                all_feat, ids, last_layer_attn)
            last_atten_pt = last_atten_pt[:, 1:].contiguous()
        else:
            sents = [[self.idxtoword[w.item()] for w in sent]
                     for sent in ids]

        if self.last_n_layers == 1:
            all_feat = all_feat[:, 0]

        report_feat = all_feat[:, 0].contiguous()
        word_feat = all_feat[:, 1:].contiguous()

        # sent_feat = all_feat.mean(axis=2)
        # if self.aggregate_method == "sum":
        #     word_feat = all_feat.sum(axis=1)
        #     sent_feat = sent_feat.sum(axis=1)
        # elif self.aggregate_method == "mean":
        #     word_feat = all_feat.mean(axis=1)
        #     sent_feat = sent_feat.mean(axis=1)
        # else:
        #     print(self.aggregate_method)
        #     raise Exception("Aggregation method not implemented")

        # aggregate intermetidate layers
        # TODO: try to remove word later
        # if self.last_n_layers > 1:
        #     all_feat = torch.stack(
        #         all_feat[-self.last_n_layers:]
        #     )  # layers, batch, sent_len, embedding size

        #     all_feat = all_feat.permute(1, 0, 2, 3)

        #     if self.agg_tokens:
        #         all_feat, sents = self.aggregate_tokens(all_feat, ids)
        #     else:
        #         sents = [[self.idxtoword[w.item()] for w in sent]
        #                  for sent in ids]
        #     sent_feat = all_feat.mean(axis=2)

        #     if self.aggregate_method == "sum":
        #         word_feat = all_feat.sum(axis=1)
        #         sent_feat = sent_feat.sum(axis=1)
        #     elif self.aggregate_method == "mean":
        #         word_feat = all_feat.mean(axis=1)
        #         sent_feat = sent_feat.mean(axis=1)
        #     else:
        #         print(self.aggregate_method)
        #         raise Exception("Aggregation method not implemented")
        # else:
        #     # use last layer
        #     word_feat, sent_feat = outputs[0], outputs[1]
        # word_feat = rearrange(word_feat, "b n d -> b d n")

        # if get_local:
        #     return word_feat, sents, sent_indices
        # else:
        #     return sents, sent_indices

        # report_feat, report_atten_weights = self.atten_pooling(sent_feat)

        # use cls token as report features
        # report_feat = word_feat[:, 0].contiguous()
        # use mean here

        return report_feat, word_feat, last_atten_pt, sents

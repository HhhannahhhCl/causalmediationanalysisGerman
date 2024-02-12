
import math
import statistics
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
    #GPT2LMHeadModel, GPT2Tokenizer
)


from attention_intervention_model import (
    AttentionOverride
)
from utils import batch, convert_results_to_pd

np.random.seed(1)
torch.manual_seed(1)



class Intervention():
    '''
    Wrapper for all the possible interventions
    '''
    def __init__(self,
                 tokenizer,
                 base_string: str,
                 substitutes: list,
                 candidates: list,
                 device='cpu'):
        super()
        self.device = device
        self.enc = tokenizer

        ''' NOT NEEDED!
        if isinstance(tokenizer, XLNetTokenizer):
            base_string = PADDING_TEXT + ' ' + base_string
        '''

        # All the initial strings
        # First item should be neutral, others tainted
        self.base_strings = [base_string.format(s)
                             for s in substitutes]
        # Tokenized bases
        self.base_strings_tok = [
            self.enc.encode(s,
                            add_special_tokens=False) # , add_space_before_punct_symbol=True) # does not exist for german gpt2 tokenizer
                            # experiments with english gpt2 show no difference in tokenization, this seems to be rather relevant for TransformerXL
            for s in self.base_strings
        ]
        # print(self.base_strings_tok)
        self.base_strings_tok = torch.LongTensor(self.base_strings_tok)\
                                     .to(device)
        # Where to intervene
        self.position = base_string.split().index('{}')

        self.candidates = []
        for c in candidates:
            # print(f'Candidate: {c}')
            # 'a ' added to input so that tokenizer understand that first word follows a space. 
            # --> tokenization will lead to Ġ infront of first 'real' token since that is the encoding of a space
            # be aware that ä, ö, ü, and ß will be encoded with special characters but decoded correctly
            tokens = self.enc.tokenize(
                'ein ' + c                                #'a ' + c,
                )[1:]    # add_space_before_punct_symbol=True
            # print(tokens)
            self.candidates.append(tokens)

        self.candidates_tok = [self.enc.convert_tokens_to_ids(tokens)
                               for tokens in self.candidates]


class Model():
    '''
    Wrapper for all model logic
    '''
    def __init__(self,
                 device='cpu',
                 output_attentions=True,
                 random_weights=False,
                 #masking_approach=1,
                 gpt2_version='gpt2'):
        super()

        #self.is_gpt2 = (gpt2_version.startswith('gpt2') or
        #                gpt2_version.startswith('distilgpt2'))
        #assert (self.is_gpt2) 

        self.device = device
        gpt2_config = GPT2Config.from_pretrained(gpt2_version)
        gpt2_config.output_attentions = True
        self.model = (AutoModelForCausalLM).from_pretrained( # AutoModelForCausalLM
                      #GPT2LMHeadModel if self.is_gpt2 else
            gpt2_version,
            #output_attentions = output_attentions)
            config=gpt2_config)
        self.gpt2_config = gpt2_config
        self.model.eval()
        self.model.to(device)
        if random_weights:
            print('Randomizing weights')
            self.model.init_weights()

        # Options
        self.top_k = 5
        self.num_layers = self.model.config.num_hidden_layers
        self.num_neurons = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        # self.masking_approach = masking_approach # Used only for masked LMs
        # assert masking_approach in [1, 2, 3, 4, 5, 6]

        tokenizer = (AutoTokenizer).from_pretrained(gpt2_version)
                     #GPT2Tokenizer if self.is_gpt2 else
        # Special token id's: (mask, cls, sep)
        self.st_ids = (tokenizer.mask_token_id,
                       tokenizer.cls_token_id,
                       tokenizer.sep_token_id)

        # To account for switched dimensions in model internals:
        # Default: [batch_size, seq_len, hidden_dim],
        # txl and xlnet: [seq_len, batch_size, hidden_dim]
        self.order_dims = lambda a: a

        self.attention_layer = lambda layer: self.model.transformer.h[layer].attn
        self.word_emb_layer = self.model.transformer.wte
        self.neuron_layer = lambda layer: self.model.transformer.h[layer].mlp


    def get_representations(self, context, position):
        # Hook for saving the representation
        def extract_representation_hook(module,
                                        input,
                                        output,
                                        position,
                                        representations,
                                        layer):
            representations[layer] = output[self.order_dims((0, position))]
        handles = []
        representation = {}
        with torch.no_grad():
            # construct all the hooks
            # word embeddings will be layer -1
            handles.append(self.word_emb_layer.register_forward_hook(
                partial(extract_representation_hook,
                        position=position,
                        representations=representation,
                        layer=-1)))
            # hidden layers
            for layer in range(self.num_layers):
                handles.append(self.neuron_layer(layer).register_forward_hook(
                    partial(extract_representation_hook,
                            position=position,
                            representations=representation,
                            layer=layer)))
            self.model(context.unsqueeze(0))
            for h in handles:
                h.remove()
        # print(representation[0][:5])
        return representation

    def get_probabilities_for_examples(self, context, candidates):
        """Return probabilities of single-token candidates given context"""
        for c in candidates:
            if len(c) > 1:
                raise ValueError(f"Multiple tokens not allowed: {c}")
        outputs = [c[0] for c in candidates]
        logits = self.model(context)[0]
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        return probs[:, outputs].tolist()

    def get_probabilities_for_examples_multitoken(self, context, candidates):
        """
        Return probability of multi-token candidates given context.
        Prob of each candidate is normalized by number of tokens.

        Args:
            context: Tensor of token ids in context
            candidates: list of list of token ids in each candidate

        Returns: list containing probability for each candidate
        """
        # TODO: Combine into single batch
        mean_probs = []
        context = context.tolist()
        for candidate in candidates:
            token_log_probs = []
            combined = context + candidate
            # Exclude last token position when predicting next token
            batch = torch.tensor(combined[:-1]).unsqueeze(dim=0).to(self.device)
            # Shape (batch_size, seq_len, vocab_size)
            #print(np.shape(batch))
            #print(batch)
            logits = self.model(batch)[0]
            # Shape (seq_len, vocab_size)
            log_probs = F.log_softmax(logits[-1, :, :], dim=-1)
            context_end_pos = len(context) - 1
            continuation_end_pos = context_end_pos + len(candidate)
            # TODO: Vectorize this
            # Up to but not including last token position
            for i in range(context_end_pos, continuation_end_pos):
                next_token_id = combined[i+1]
                next_token_log_prob = log_probs[i][next_token_id].item()
                token_log_probs.append(next_token_log_prob)
            mean_token_log_prob = statistics.mean(token_log_probs)
            mean_token_prob = math.exp(mean_token_log_prob)
            mean_probs.append(mean_token_prob)
        return mean_probs

    #def neuron_intervention():
        # Hook for changing representation during forward pass
        #def intervention_hook()

    def head_pruning_intervention(self,
                                  context,
                                  outputs,
                                  layer,
                                  head):
        # Recreate model and prune head
        save_model = self.model
        # TODO Make this more efficient
        self.model = AutoModelForCausalLM.from_pretrained(gpt2_version) #('gpt2')
        self.model.prune_heads({layer: [head]})
        self.model.eval()

        # Compute probabilities without head
        new_probabilities = self.get_probabilities_for_examples(
            context,
            outputs)

        # Reinstate original model
        # TODO Handle this in cleaner way
        self.model = save_model

        return new_probabilities

    def attention_intervention(self,
                               context,
                               outputs,
                               attn_override_data):
        """ Override attention values in specified layer

        Args:
            context: context text
            outputs: candidate outputs
            attn_override_data: list of dicts of form:
                {
                    'layer': <index of layer on which to intervene>,
                    'attention_override': <values to override the computed attention weights.
                           Shape is [batch_size, num_heads, seq_len, seq_len]>,
                    'attention_override_mask': <indicates which attention weights to override.
                                Shape is [batch_size, num_heads, seq_len, seq_len]>
                }
        """

        def intervention_hook(module, input, outputs, attn_override, attn_override_mask, gpt2_config):
            attention_override_module = (AttentionOverride)(
                module, attn_override, attn_override_mask, gpt2_config) #self.gpt2_config
            return attention_override_module(*input)

        with torch.no_grad():
            hooks = []
            for d in attn_override_data:
                attn_override = d['attention_override']
                # print(np.shape(attn_override))
                attn_override_mask = d['attention_override_mask']
                # print(np.shape(attn_override_mask))
                layer = d['layer']
                hooks.append(self.attention_layer(layer).register_forward_hook(
                    partial(intervention_hook,
                            attn_override=attn_override,
                            attn_override_mask=attn_override_mask,
                            gpt2_config = self.gpt2_config)))

            new_probabilities = self.get_probabilities_for_examples_multitoken(
                context,
                outputs)

            for hook in hooks:
                hook.remove()

            return new_probabilities

    #def neuron_intervention_experiment()

    #def neuron_intervention_single_experiment()

    def attention_intervention_experiment(self, intervention, effect):
        """
        Run one full attention intervention experiment
        measuring indirect or direct effect.
        """
        # E.g. The doctor asked the nurse a question. He
        x = intervention.base_strings_tok[0]
        print(f'x = {x}')
        # E.g. The doctor asked the nurse a question. She
        x_alt = intervention.base_strings_tok[1]
        print(f'x_alt = {x_alt}')

        if effect == 'indirect':
            input = x_alt  # Get attention for x_alt
        elif effect == 'direct':
            input = x  # Get attention for x
        else:
            raise ValueError(f"Invalid effect: {effect}")
        batch = input.clone().detach().unsqueeze(0).to(self.device)
        attention_override = self.model(batch)[-1] # tuple of 12 layers; attention of either x_alt (NIE) or x (NDE) for whole model, used below
        # each element (batch_size, num_heads in layer (12), seq_len, seq_len)
        # print(attention_override)

        batch_size = 1
        seq_len = len(x)
        seq_len_alt = len(x_alt)
        assert seq_len == seq_len_alt

        with torch.no_grad():

            candidate1_probs_head = torch.zeros((self.num_layers, self.num_heads))
            candidate2_probs_head = torch.zeros((self.num_layers, self.num_heads))
            candidate1_probs_layer = torch.zeros(self.num_layers)
            candidate2_probs_layer = torch.zeros(self.num_layers)

            if effect == 'indirect':
                context = x # Use context of x while having attention for x_alt
            else:
                context = x_alt # Use context of x_alt while having attention of x

            # Intervene at every layer and head by overlaying attention induced by x_alt (NIE) or x (NDE)
            model_attn_override_data = [] # Save layer interventions for model-level intervention later
            # level of layers
            for layer in range(self.num_layers):
                layer_attention_override = attention_override[layer] # torch.tensor(np.array(attention_override[layer]))
                print(type(layer_attention_override), np.shape(layer_attention_override))
                #print(type(attention_override), np.shape(attention_override),
                #    type(torch.tensor(np.array(layer_attention_override))), np.shape(torch.tensor(np.array(layer_attention_override))))
                # activate masking all heads in layer by setting value to 1
                attention_override_mask = torch.ones_like(layer_attention_override, dtype=torch.uint8) 
                layer_attn_override_data = [{
                    'layer': layer,
                    'attention_override': layer_attention_override, # attn_override; shape: (2, 1, 12, 11, 64)
                    'attention_override_mask': attention_override_mask
                }]
                candidate1_probs_layer[layer], candidate2_probs_layer[layer] = self.attention_intervention(
                    context=context,
                    outputs=intervention.candidates_tok,
                    attn_override_data = layer_attn_override_data)
                model_attn_override_data.extend(layer_attn_override_data)
                # level of heads
                for head in range(self.num_heads): 
                    attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.uint8)
                    attention_override_mask[0][head] = 1 # Set mask to 1 for single head only
                    head_attn_override_data = [{
                        'layer': layer,
                        'attention_override': layer_attention_override,
                        'attention_override_mask': attention_override_mask
                    }]
                    candidate1_probs_head[layer][head], candidate2_probs_head[layer][head] = self.attention_intervention(
                        context=context,
                        outputs=intervention.candidates_tok,
                        attn_override_data=head_attn_override_data) # per head; list of dicts

            # Intervene on entire model by overlaying attention induced by x_alt
            candidate1_probs_model, candidate2_probs_model = self.attention_intervention(
                context=context,
                outputs=intervention.candidates_tok,
                attn_override_data=model_attn_override_data) # layer interventions for whole model
                # one element in list = one layer, each element dict

        return candidate1_probs_head, candidate2_probs_head, candidate1_probs_layer, candidate2_probs_layer,\
            candidate1_probs_model, candidate2_probs_model

    def attention_intervention_single_experiment(self, intervention, effect, layers_to_adj, heads_to_adj, search):
        """
        Run one full attention intervention experiment
        measuring indirect or direct effect.
        """
        # E.g. The doctor asked the nurse a question. He
        x = intervention.base_strings_tok[0]
        # E.g. The doctor asked the nurse a question. She
        x_alt = intervention.base_strings_tok[1]

        if effect == 'indirect':
            input = x_alt  # Get attention for x_alt
        elif effect == 'direct':
            input = x  # Get attention for x
        else:
            raise ValueError(f"Invalid effect: {effect}")
        batch = torch.tensor(input).unsqueeze(0).to(self.device)
        attention_override = self.model(batch)[-1]

        batch_size = 1
        seq_len = len(x)
        seq_len_alt = len(x_alt)
        assert seq_len == seq_len_alt
        assert len(attention_override) == self.num_layers
        assert attention_override[0].shape == (batch_size, self.num_heads, seq_len, seq_len)

        with torch.no_grad():
            if search:
                candidate1_probs_head = torch.zeros((self.num_layers, self.num_heads))
                candidate2_probs_head = torch.zeros((self.num_layers, self.num_heads))

            if effect == 'indirect':
                context = x
            else:
                context = x_alt

            model_attn_override_data = []
            for layer in range(self.num_layers):
                if layer in layers_to_adj:
                    layer_attention_override = attention_override[layer]

                    layer_ind = np.where(layers_to_adj == layer)[0]
                    heads_in_layer = heads_to_adj[layer_ind]
                    attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.uint8)
                    # set multiple heads in layer to 1
                    for head in heads_in_layer:
                        attention_override_mask[0][head] = 1 # Set mask to 1 for single head only
                    # get head mask
                    head_attn_override_data = [{
                        'layer': layer,
                        'attention_override': layer_attention_override,
                        'attention_override_mask': attention_override_mask
                    }]
                    # should be the same length as the number of unique layers to adj
                    model_attn_override_data.extend(head_attn_override_data)

            # basically generate the mask for the layers_to_adj and heads_to_adj
            if search:
                for layer in range(self.num_layers):
                  layer_attention_override = attention_override[layer]
                  layer_ind = np.where(layers_to_adj == layer)[0]
                  heads_in_layer = heads_to_adj[layer_ind]

                  for head in range(self.num_heads):
                    if head not in heads_in_layer:
                          model_attn_override_data_search = []
                          attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.uint8)
                          heads_list = [head]
                          if len(heads_in_layer) > 0:
                            heads_list.extend(heads_in_layer)
                          for h in (heads_list):
                              attention_override_mask[0][h] = 1 # Set mask to 1 for single head only
                          head_attn_override_data = [{
                              'layer': layer,
                              'attention_override': layer_attention_override,
                              'attention_override_mask': attention_override_mask
                          }]
                          model_attn_override_data_search.extend(head_attn_override_data)
                          for override in model_attn_override_data:
                              if override['layer'] != layer:
                                  model_attn_override_data_search.append(override)

                          candidate1_probs_head[layer][head], candidate2_probs_head[layer][head] = self.attention_intervention(
                              context=context,
                              outputs=intervention.candidates_tok,
                              attn_override_data=model_attn_override_data_search)
                    else:
                        candidate1_probs_head[layer][head] = -1
                        candidate2_probs_head[layer][head] = -1


            else:
              candidate1_probs_head, candidate2_probs_head = self.attention_intervention(
                  context=context,
                  outputs=intervention.candidates_tok,
                  attn_override_data=model_attn_override_data)

        return candidate1_probs_head, candidate2_probs_head

# This is a neuron intervention!
def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoModelForCausalLM.from_pretrained('gpt2') 
    model = Model(device=DEVICE)

    base_sentence = "The {} said that"
    biased_word = "teacher"
    intervention = Intervention(
            tokenizer,
            base_sentence,
            [biased_word, "man", "woman"],
            ["he", "she"],
            device=DEVICE)
    interventions = {biased_word: intervention}

    intervention_results = model.neuron_intervention_experiment(
        interventions, 'man_minus_woman')
    df = convert_results_to_pd(
        interventions, intervention_results)
    print('more probable candidate per layer, across all neurons in the layer')
    print(df[0:5])


if __name__ == "__main__":
    main()

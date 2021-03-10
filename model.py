
"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import DistilBertModel
from transformers import DistilBertPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from transformers.configuration_utils import PretrainedConfig


class MoEbertConfig():
    """ base GPT config, params common to all GPT versions """
    flavor = "distilbert-base-uncased"
    num_experts = 12
    expert_hidden_size = 1024
    qa_pdrop = 0.1
    max_length=384
    weight_importance=1e-2
    num_init = 0
    exp_crossreg = 0


    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.fc2(self.relu(self.fc1(X)))

class MoEbert(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        self.distBert = DistilBertModel.from_pretrained(config.flavor)
        self.experts = nn.ModuleList([MLP(768, config.expert_hidden_size, 768) for i in range(config.num_experts)])

        self.gateNet = nn.Linear(768 * config.max_length, config.num_experts)
        self.softmax = nn.Softmax(dim=1)

        self.qa = nn.Linear(768, 2)
        self.dropout = nn.Dropout(config.qa_pdrop)
        self.w_imp = config.weight_importance
        self.exp_crossreg = config.exp_crossreg
        self.num_weights_per_expert = torch.cat((self.experts[0].fc1.weight.flatten(),self.experts[0].fc2.weight.flatten())).shape[0]
        for i in range(config.num_init):
            self.distBert.transformer.layer[self.distBert.transformer.n_layers - 1 - i].apply(self.distBert._init_weights)
        print("number of parameters: {}".format(sum(p.numel() for p in self.parameters())))


    def forward(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                start_positions=None,
                end_positions=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        distOut = self.distBert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict) #last hidden state (batch_size, max_len, 768)

        enc = distOut[0]
        enc = self.dropout(enc)

        #print(exp_outs[0].shape)
        gate = self.softmax(self.gateNet(torch.flatten(enc, start_dim=1))) #b x num_experts

        ###V1
        # Y = torch.cuda.FloatTensor(enc.shape).fill_(0)
        # exp_outs = [exp(enc) for exp in self.experts] #num_experts x 768
        #
        # for i, out in enumerate(exp_outs):
        #     Y += gate[:,i,None,None] * out

        ###V2

        exp_outs = [torch.unsqueeze(exp(enc),1) for exp in self.experts]
        exp_outs = torch.cat(exp_outs, 1)
        expanded_gate = gate.unsqueeze(2).unsqueeze(3).expand_as(exp_outs)
        Y = torch.sum(torch.mul(exp_outs, expanded_gate), dim=1).squeeze(1)

        logits = self.qa(Y)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (batch_size, max_len)
        end_logits = end_logits.squeeze(-1)

        I = torch.sum(gate, dim=0)
        if I.shape[0] == 1:
            I_cv = I[0]
        else:
            I_cv = I.float().var() / (I.float().mean()**2 + 1e-10)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            total_loss += I_cv * self.w_imp


            if self.exp_crossreg > 0 :
                t = torch.empty((len(self.experts), self.num_weights_per_expert), device=torch.cuda.current_device())
                for i, exp in enumerate(self.experts):
                    t[i,:] = torch.cat((exp.fc1.weight.flatten(),exp.fc2.weight.flatten()))
                t_var = torch.var(t, dim=0)
                total_loss += 1/self.exp_crossreg*(t_var.sum()/self.num_weights_per_expert) * self.exp_crossreg


            #if not return_dict:
         #   output = (start_logits, end_logits) + distOut[1:]
         #   return ((total_loss,) + output) if total_loss is not None else output


        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distOut.hidden_states,
            attentions=distOut.attentions,
        )


##Only for eval!
class EnsembleMoE(nn.Module):
    def __init__(self, modelList):
        self.modelList = modelList

    def forward(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                start_positions=None,
                end_positions=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        start_logits = None
        end_logits = None

        for model in self.modelList:
            outputs = (model(input_ids=None,
                                attention_mask=None,
                                head_mask=None,
                                inputs_embeds=None,
                                start_positions=None,
                                end_positions=None,
                                output_attentions=None,
                                output_hidden_states=None,
                                return_dict=None))
            if start_logits is None:
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
            else:
                start_logits += outputs.start_logits
                end_logits += outputs.end_logits


        return QuestionAnsweringModelOutput(
            loss = None,
            start_logits = start_logits,
            end_logits=end_logits,
            hidden_states = None,
            attentions=None
        )


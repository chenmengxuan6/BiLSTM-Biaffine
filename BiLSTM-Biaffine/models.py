import torch
import torch.nn as nn
import torch.functional as F
from torchcrf import CRF

from pytorch_transformers import BertPreTrainedModel, BertModel
from biaffine import Biaffine
from biaffine import Attn
class BERT_BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)
        
        self.num_tags = config.num_labels
        self.bert = BertModel(config)
        self.wlstm = nn.LSTM(config.hidden_size, 768, num_layers=1, bidirectional=False, batch_first=True)
        # self.wlstm = nn.LSTM(config.hidden_size, 768, num_layers=1, bidirectional=True, batch_first=True)
        self.biaffine = Biaffine(768)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.need_birnn = need_birnn

        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim*2
        
        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, step, len_data, tags, token_type_ids=None, input_mask=None, sentences_word=None):
        emissions = self.tag_outputs(input_ids, step, len_data, token_type_ids, input_mask,sentences_word)
        loss = -1*self.crf(emissions, tags, mask=input_mask.byte())

        return loss

    
    def tag_outputs(self, input_ids, step, len_data, token_type_ids=None, input_mask=None, sentences_word=None):

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        sequence_output = outputs[0]  

        biaffine_sent_outputs = torch.tensor([]).to(sequence_output.device)
        w = 0
        w_start = step * 8
        sl = torch.zeros(1,1,768).to(sequence_output.device)
        for sentence in sentences_word[w_start:w_start + (8 if step <= (len_data+1) // 8 else (len_data+1) % 8)]:
            start = 0
            biaffine_sent_output = torch.tensor([]).to(sequence_output.device)
            for word in sentence: 
                _,h= self.wlstm(sequence_output[w,start:start+word].unsqueeze(0))
                wlstm_output_forword = h[0]
                wlstm_output_backword = h[1]
                biaffine_output = self.biaffine(wlstm_output_forword,wlstm_output_backword)
                biaffine_sent_output = torch.cat((biaffine_sent_output, biaffine_output),1)
                start = start + word
            while biaffine_sent_output.size()[1] < 256:  
                biaffine_sent_output = torch.cat((biaffine_sent_output,sl),1)
            biaffine_sent_outputs = torch.cat((biaffine_sent_outputs, biaffine_sent_output), 0)
            w = w + 1
        if self.need_birnn:
            sequence_output, _ = self.birnn(biaffine_sent_outputs)

        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)

        return emissions
    
    def predict(self, input_ids, step, len_data,token_type_ids=None, input_mask=None, sentences_word=None):
        emissions = self.tag_outputs(input_ids, step, len_data, token_type_ids, input_mask, sentences_word)
        return self.crf.decode(emissions, input_mask.byte())

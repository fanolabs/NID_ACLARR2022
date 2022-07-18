"""
modified from
https://github.com/thuiar/DeepAligned-Clustering/blob/main/model.py
"""

from utils.tools import *
from utils.contrastive import SupConLoss
        
class BertForModel(nn.Module):
    def __init__(self,model_name, num_labels, device=None):
        super(BertForModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.device = device
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.classifier = nn.Linear(768, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.backbone.to(self.device)
        self.classifier.to(self.device)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        """logits are not normalized by softmax in forward function"""
        outputs = self.backbone(**X, output_hidden_states=True)
        # extract last layer [CLS]
        CLSEmbedding = outputs.hidden_states[-1][:,0]
        CLSEmbedding = self.dropout(CLSEmbedding)
        logits = self.classifier(CLSEmbedding)
        output_dir = {"logits": logits}
        if output_hidden_states:
            output_dir["hidden_states"] = outputs.hidden_states[-1][:, 0]
        if output_attentions:
            output_dir["attentions"] = outputs.attention
        return output_dir

    def mlmForward(self, X, Y):
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output
    
    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)


class CLBert(nn.Module):
    def __init__(self,model_name, device, feat_dim=128):
        super(CLBert, self).__init__()
        self.model_name = model_name
        self.device = device
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, feat_dim)
        )
        self.backbone.to(self.device)
        self.head.to(device)
        
    def forward(self, X, output_hidden_states=False, output_attentions=False, output_logits=False):
        """logits are not normalized by softmax in forward function"""
        outputs = self.backbone(**X, output_hidden_states=True, output_attentions=True)
        cls_embed = outputs.hidden_states[-1][:,0]
        features = F.normalize(self.head(cls_embed), dim=1)
        output_dir = {"features": features}
        if output_hidden_states:
            output_dir["hidden_states"] = cls_embed
        if output_attentions:
            output_dir["attentions"] = outputs.attentions
        return output_dir

    def loss_cl(self, embds, label=None, mask=None, temperature=0.07, base_temperature=0.07):
        """compute contrastive loss"""
        loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        output = loss(embds, labels=label, mask=mask)
        return output
    
    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)

import torch
import torch.nn as nn
from .transformer.encoder import TransformerEncoder

class mlp_emotion_classifier(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        

        self.out = nn.Sequential(nn.Linear(115,256),
                                 nn.LeakyReLU(0.05),
                                 nn.Linear(256,256),
                                 nn.LeakyReLU(0.05),
                                 nn.Linear(256, 7))

        self.loss_func = nn.CrossEntropyLoss()

    def get_loss(
        self,
        logits,
        emotion,
    ):
        bs = logits.shape[0]
        loss = self.loss_func(logits, emotion)

        preds = torch.argmax(logits, dim = -1)
        accuracy = (preds == emotion).float().mean().detach()


        return loss, {      'CELoss': {
                                'val': loss.item(),
                                'n': bs,
                                'weight':1
                            },
                            'Acc':{
                                'val': accuracy.item(),
                                'n': bs,
                                'weight':1
                            }
                    }

    
    def forward(
        self,
        motion_coef,
        emotion = None
    ):
        if len(motion_coef.shape)>=3:
            motion_coef = motion_coef.squeeze(1)
        if len(emotion.shape)>=2:
            emotion = emotion.squeeze(1)

        logits = self.out(motion_coef)
        #print(logits.size())
        #print(emotion.size())
        if emotion is not None:
            loss, loss_dict = self.get_loss(
                logits,
                emotion,
            )
            return loss, loss_dict, logits
        return logits


class emotion_classifier(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(input_size=115, output_size=256, linear_units=1024, input_layer='linear', num_blocks=3, dropout_rate=0.1, attention_heads=4)
        self.out = nn.Linear(256, 7)

        self.loss_func = nn.CrossEntropyLoss()

    def get_loss(
        self,
        logits,
        emotion,
    ):
        bs = logits.shape[0]
        loss = self.loss_func(logits, emotion)

        preds = torch.argmax(logits, dim = -1)
        accuracy = (preds == emotion).float().mean().detach()


        return loss, {      'CELoss': {
                                'val': loss.item(),
                                'n': bs,
                                'weight':1
                            },
                            'Acc':{
                                'val': accuracy.item(),
                                'n': bs,
                                'weight':1
                            }
                    }
  
    def forward(
        self,
        motion_coef,
        lengths,
        emotion = None
    ):
        encoder_out, encoder_mask, pos = self.encoder(motion_coef, lengths)
        
        b,t,d = encoder_out.shape
        mask = torch.arange(t).expand(b, t) < lengths.unsqueeze(1)
        masked_features = encoder_out * mask.unsqueeze(-1) 
        sum_features = masked_features.sum(dim=1) 
        valid_lengths = lengths.clamp(min=1).unsqueeze(-1)
        avg_features = sum_features / valid_lengths             #

        logits = self.out(avg_features)
        
        if emotion is not None:

            loss, loss_dict = self.get_loss(
                logits,
                emotion,
            )

            return loss, loss_dict, logits
        return logits

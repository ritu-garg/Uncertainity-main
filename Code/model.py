import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViTModel,AutoImageProcessor



ViT_MODEL = "google/vit-base-patch16-224-in21k"
IMG_EMBED_DIM = 768
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImgEncoder(nn.Module):
    # define model elements
    def __init__(self):
        super(ImgEncoder, self).__init__()
        #self.image_processor = AutoImageProcessor.from_pretrained(ViT_MODEL)
        self.vit_encoder = ViTModel.from_pretrained(ViT_MODEL)

        for p in self.vit_encoder.parameters():
            p.requires_grad = True

    def forward(self, image):
        ''' forward func  .. .'''
        ''' img embedding layer '''
        #inputs = self.image_processor(image)
        outputs = self.vit_encoder(image)
        return outputs.pooler_output
        #return outputs.pooled_output


class Neural_Img_Clf_model(nn.Module):
    # define model elements
    def __init__(self, no_classes=100,dropout=0.2):
        super(Neural_Img_Clf_model, self).__init__()

        self.img_encoder = ImgEncoder()
        self.projection = nn.Sequential(nn.Dropout(dropout,inplace=False),nn.Linear(IMG_EMBED_DIM, no_classes))
        # self.projection = nn.Linear(IMG_EMBED_DIM, no_classes)

    # forward propagate input
    def forward(self, img_feat_in, mode='embed'):
        ''' forward func '''
        img_embed = self.img_encoder(img_feat_in)
        #print(img_embed.shape)
        return img_embed if mode == 'embed' else self.projection(img_embed)



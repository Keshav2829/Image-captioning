from core.model.vgg16_encoder import VGG16
from core.model.lstm_decoder import LSTMDecoderWithProjectLayer
from torchsummary import summary
import yaml
import torch
from core.data_prep.text_preprocessing import ProcessData, TrainTokenizer
from dataloader import ImageCaptionDataset,collate_fuction
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
from training import Train
import numpy as np
from PIL import Image, ImageDraw


def showImg(img, caption):
    img = img.numpy()
    img*= 255
    img = img.astype(np.uint8)
    img = img.squeeze().transpose(1,2,0)
    img = Image.fromarray(img)
    I1 = ImageDraw.Draw(img)
    I1.text((28, 36), caption, fill=(255, 0, 0))
    img.show()
# prep = ProcessData('data/dataset/flicker8k/captions.txt')
# # prep.getImageEmbeddings('data/dataset/flicker8k')
# prep.save_data('data/dataset/flicker8k/captions_processed.csv')
# # train,test = prep.test_train_split()
# test_data_path = 'data/dataset/flicker8k/captions_test.csv'
test_data_path = 'data/dataset/flicker8k/captions_test.csv'
train_data_path = 'data/dataset/flicker8k/captions_train.csv'

# train,test = prep.test_train_split(train_data_path, test_data_path)
# # data_path = "data/dataset/flicker8k/captions_processed.csv"
tokensizer = AutoTokenizer.from_pretrained('data/tokenizer')
img_transforms = transforms.Compose([
    transforms.Resize((224,224))
])
# data = ImageCaptionDataset(data_path=train_data_path, root_dir='data/dataset/flicker8k', tranforms=img_transforms)
# dataloader = DataLoader(data, batch_size=2, collate_fn= collate_fuction(tokensizer))
# img,input, target = next(iter(dataloader))
# tokenizer_train = TrainTokenizer(data_path=data_path)
# tokenizer_train.train()
# tokenizer_train.save('data/tokenizer')
trainer = Train()
# trainer.train()
# print(tokensizer.tokenize(caption))

data = ImageCaptionDataset(data_path=test_data_path, root_dir='data/dataset/flicker8k', tranforms=img_transforms)
dataloader = DataLoader(data, batch_size=1, collate_fn= collate_fuction(tokensizer), shuffle=True)
img,input, target = next(iter(dataloader))
caption = trainer.predict(img)
showImg(img, caption)
# print(tokensizer.decode(target))



print(caption)

# class AttributeDict(dict):
#     def __getattr__(self, attr):
#         return self[attr]
#     def __setattr__(self, attr, value):
#         self[attr] = value

# encoder_model = VGG16().to('cuda')
# print(encoder_model)
# encoder_model.load_state_dict(torch.load('./data/weights/VGG_16/weights.pth', weights_only=True))
# print(summary(encoder_mod
# el, (3,224,224), batch_size=-1, device='cuda'))

# decoder_config = AttributeDict(yaml.safe_load(open('config.yaml')))
# print(decoder_config)
# decoder_model = LSTMDecoderWithProjectLayer(decoder_config).to('cuda')
# input_x = torch.randint(0,5000, (10,)).to('cuda')
# img_embeddings = torch.randn(2, 4096).to('cuda')
# output = decoder_model(input_x, img_embeddings)
# print(output.shape)
# print(summary(decoder_model, (10,),input_type='specific',batch_size=-1, device='cuda'))
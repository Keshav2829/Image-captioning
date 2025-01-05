import pandas as pd
import torch.utils
import torch.utils.data
from torchtext.data import get_tokenizer
from transformers import AutoTokenizer
from core.model.vgg16_encoder import VGG16
import torch
import os
from torchvision.io import read_image
from torchvision import transforms
import json
from tqdm import tqdm

class ProcessData:
    def __init__(self, path):
        # with open(path, 'r') as file:
        #         captions = file.readlines()

        self.captions = pd.read_csv(path)
        self.captions['caption'] =  self.captions['caption'].map(self.preprocess_data)
        self.tokenizer = get_tokenizer('basic_english')   

        # self.captions.map(self.preprocess_data)
        # random shuffle
        # self.captions = self.captions.sample(frac=1).reset_index(drop=True)
        # self.vocab_size = self.get_vocab_size()
    
    def preprocess_data(self, caption):
        # caption = row['caption']
        caption = caption.lower()
        # delete digits, special chars, etc., 
        caption = caption.replace('[^A-Za-z]', '')
        # delete additional spaces
        caption = caption.replace('\s+', ' ')
        # add start and end tags to the caption
        caption = '<SoS> ' + " ".join([word for word in caption.split() if len(word)>1]) + ' <EoS>'
        # row['caption'] = caption
        return caption
    
    def save_data(self, path):
        self.captions.to_csv(path)
    
    
    def test_train_split(self, train_save_path=None, test_save_path = None):
        images = pd.Series(self.captions['image'].unique())
        train_data, test_data = torch.utils.data.random_split(images, [0.8, 0.2])
        train_images = images.iloc[train_data.indices]
        test_images = images.iloc[test_data.indices]

        train_df : pd.DataFrame = self.captions[self.captions['image'].isin(train_images)]
        test_df : pd.DataFrame = self.captions[self.captions['image'].isin(test_images)]
        if train_save_path and test_save_path:
            # train_df : pd.DataFrame = self.captions.iloc[train_data.indices]
            # test_df : pd.DataFrame = self.captions.iloc[test_data.indices]
            train_df.to_csv(train_save_path)
            test_df.to_csv(test_save_path)

        return train_df, test_df
    
    # def getImageEmbeddings(self, rootdir):
    #     vgg_model = VGG16()
    #     encoder = torch.nn.Sequential(*list(vgg_model.children())[:-2])
    #     img_transform = transforms.Compose([transforms.Resize((224,224))])
    #     encoder.to('cuda')
    #     mapping ={}
    #     for _, row in tqdm(self.captions.iterrows()):
    #         image = row['image']
    #         if image not in mapping:
    #             path = os.path.join(rootdir, 'Images', image)
    #             img = read_image(path)
    #             img = img_transform(img).unsqueeze(dim=0)
    #             img = img.to(dtype=torch.float32)/255.0
    #             img = img.to('cuda')
    #             with torch.no_grad():
    #                 embeddings = encoder(img)
                
    #             mapping[image] = embeddings.cpu().detach().numpy().tolist()
        
    #     with open("data/dataset/embedding_mapping.json", 'w') as file:
    #         json.dump(mapping, file)






class TrainTokenizer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        # self.base_tokensizer = AutoTokenizer.from_pretrained('gpt2')

    
    def get_training_corpus(self):
        for start_idx in range(0, len(self.data), 1000):
            samples = self.data.iloc[start_idx: start_idx+1000]
            yield samples['caption']

    def train(self):
        base_tokensizer = AutoTokenizer.from_pretrained('gpt2')
        self.new_tokensizer= base_tokensizer.train_new_from_iterator(self.get_training_corpus(), 5000)
        self.new_tokensizer.add_special_tokens({'eos_token' : '<EoS>', 'bos_token':'<SoS>','unk_token':'<Unk>', 'pad_token':'<EoS>'})
    

    def save(self, path):
        self.new_tokensizer.save_pretrained(path)





    


    

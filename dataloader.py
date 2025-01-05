from torch.utils.data import Dataset, dataloader
from torchvision.io import read_image
import pandas as pd
import os
import torch
from transformers import AutoTokenizer

class ImageCaptionDataset(Dataset):
    def __init__(self, data_path, root_dir, tranforms=None):
        
        self.data : pd.DataFrame = pd.read_csv(data_path)
        self.transforms = tranforms
        self.image_root = os.path.join(root_dir, "Images")

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = read_image(os.path.join(self.image_root, sample['image']))
        if self.transforms:
            image = self.transforms(image)

        image = image.to(dtype=torch.float32)/255.0
        return image, sample['caption']
    

def collate_fuction(tokenizer):
    def collate_fn(batch):

        images, captions = zip(*batch)
        # max_caption_len = max(len(cap) for cap in captions)
        tokenize_captions = tokenizer.batch_encode_plus(captions, padding = True, return_tensors = 'pt')
        ids = tokenize_captions['input_ids']
        inputs =ids[:, :-1]
        targets = ids[:, 1:]

        return torch.stack(images, dim=0), inputs, targets

    
    return collate_fn
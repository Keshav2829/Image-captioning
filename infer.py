import torch
from PIL import Image
from torchvision import transforms
import torchvision
import yaml
from core.model.lstm_decoder import LSTMDecoderWithProjectLayer
from core.model.vgg16_encoder import VGGHead # Assuming you have these model classes defined
from transformers import AutoTokenizer

class AttributeDict(dict):
    """
    A dictionary subclass that allows attribute-style access.
    """
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

class ImageCaptioning:
    def __init__(self):
        self.config =  AttributeDict(yaml.safe_load(open('infer_config.yaml')))
        self.device = self.config.device
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer['path'])
        self.img_transforms = transforms.Compose([
                                    transforms.ToTensor(),      
                                    transforms.Resize((224,224))
                                ])
        self.n_lstm_layers = self.config.decoder['n_layers']
        self.load_models()
    
    def load_models(self):
        vgg_model = torchvision.models.vgg16()
        vgg_net = list(vgg_model.children())[:-2]

        self.encoder = torch.nn.Sequential(
            *vgg_net,
            VGGHead(self.config.encoder))
        self.decoder = LSTMDecoderWithProjectLayer(self.config, len(self.tokenizer))
        self.encoder.load_state_dict(torch.load(self.config.encoder['save_path'], map_location=self.device))
        self.decoder.load_state_dict(torch.load(self.config.decoder["save_path"], map_location=self.device))
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()
    
    def predict(self, img_path):
        img = Image.open(img_path)
        img = self.img_transforms(img)
        img = img.to(self.device)
        img = img.unsqueeze(0)
        sequence = []
        # return self.trainer.predict(img)
        with torch.no_grad():
            img_features = self.encoder(img)
            input = torch.tensor([self.tokenizer.bos_token_id]).to(self.device)
            h_n = torch.zeros(self.n_lstm_layers, 1, self.config.decoder['hidden_size']).to(self.device)
            c_n = torch.zeros(self.n_lstm_layers, 1, self.config.decoder['hidden_size']).to(self.device)
            sequence.append(input.item())
            for _ in range(20):
                input = input.unsqueeze(0)
                logits, (h_n, c_n) = self.decoder(input, img_features, h_n, c_n)
                logits = torch.softmax(logits[:, -1, :], dim=-1)
                logits = logits.squeeze(1)
                predicted_id = torch.argmax(logits, dim=-1)
                input = predicted_id
                sequence.append(input.item())
                if predicted_id == self.tokenizer.eos_token_id:
                    break
        return self.tokenizer.decode(sequence,skip_special_tokens=False)
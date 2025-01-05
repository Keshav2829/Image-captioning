import torchvision
import torch
from core.model.vgg16_encoder import VGG16, VGGHead
from core.model.lstm_decoder import LSTMDecoderWithProjectLayer
import yaml
from core.data_prep.text_preprocessing import ProcessData, TrainTokenizer
from transformers import AutoTokenizer
from dataloader import ImageCaptionDataset,collate_fuction
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class AttributeDict(dict):
    """
    A dictionary subclass that allows attribute-style access.
    """
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

class Train:
    """
    A class to handle the training process of the image captioning model.
    """
    
    def __init__(self):
        """
        Initialize the training process by loading configurations, datasets, and models.
        """
        # self.config = yaml.load(open())
        self.config = AttributeDict(yaml.safe_load(open('config.yaml')))
        ## load encoder and decoder models
        self.device = self.config.device
        self.tokensizer = AutoTokenizer.from_pretrained(self.config.tokenizer['path'])
        img_transforms = transforms.Compose([
                                    transforms.Resize((224,224)),
                                    # transforms.ToTensor()
                                ])
        dataset = ImageCaptionDataset(data_path=self.config.data['train'], root_dir= self.config.data['rootdir'], tranforms=img_transforms)
        self.dataloader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn= collate_fuction(self.tokensizer), shuffle=True)

        val_dataset = ImageCaptionDataset(data_path=self.config.data['test'], root_dir= self.config.data['rootdir'], tranforms=img_transforms)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size, collate_fn= collate_fuction(self.tokensizer), shuffle=True)

        self.n_lstm_layers = self.config.decoder['n_layers']
        self.load_models(load_model_weighs=True)
        # self.load_weights()

    def load_models(self, load_model_weighs = True):
        """
        Load the encoder and decoder models, and optionally their weights.

        Args:
            load_model_weighs (bool): Whether to load the model weights.
        """
        vgg_model = torchvision.models.vgg16()
        encoder_path=self.config.encoder["weights_path"]
        # vgg_model = VGG16()
        if encoder_path:
            vgg_model.load_state_dict(torch.load(encoder_path))
        # # truncated_clssifier = torch.nn.Sequential(*list(vgg_model.classifier.children())[:-2])
        vgg_net = list(vgg_model.children())[:-2]
        # for params in vgg_net:
        #     params.requires_grad = False

        self.encoder = torch.nn.Sequential(
            *vgg_net,
            VGGHead(self.config.encoder))
        # self.encoder =torch.nn.Sequential(
        #         *list(vgg_model.features.children()),
        #         torch.nn.Flatten(),
        #         *list(truncated_clssifier)
        # )
        

        # for params in self.encoder.parameters():
        #     params.requires_grad = False

        self.decoder = LSTMDecoderWithProjectLayer(self.config, len(self.tokensizer))

        if load_model_weighs:
            self.load_weights(decoder_path=self.config.decoder["save_path"])

        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def load_weights(self, decoder_path = None):
        """
        Load the weights for the decoder model.

        Args:
            decoder_path (str): Path to the decoder weights file.
        """
        if decoder_path and self.config.decoder['load_weights']:
            self.decoder.load_state_dict(torch.load(decoder_path))
        
        if self.config.encoder['load_weights']:
            self.encoder.load_state_dict(torch.load(self.config.encoder['save_path']))
    
    def train(self):
        """
        Train the image captioning model.
        """
        optimizer_decoder = torch.optim.Adam(params= self.decoder.parameters(), lr=self.config.lr)
        schedular_decoder = torch.optim.lr_scheduler.LinearLR(optimizer_decoder, start_factor=0.5, total_iters=5)
        optimizer_encoder = torch.optim.Adam(params=[p for p in self.encoder.parameters() if p.requires_grad], lr=self.config.lr)
        schedular_encoder = torch.optim.lr_scheduler.LinearLR(optimizer_encoder, start_factor=0.5, total_iters=5)
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in tqdm(range(self.config.epochs), desc="Training Progress"):
            with tqdm(total= len(self.dataloader), desc= f"Epoch {epoch+1}/{self.config.epochs}") as pbar:
                epoch_loss = 0
                self.decoder.train()
                for img, in_s, out_s in self.dataloader:
                    optimizer_decoder.zero_grad()
                    optimizer_encoder.zero_grad()
                    img, in_s, out_s = img.to(self.device), in_s.to(self.device), out_s.to(self.device)
                    img_embeddings = self.encoder(img)
                    # img_embeddings = torch.unsqueeze(img_embeddings, dim= 0).repeat(self.n_lstm_layers, 1, 1)
                    pred_s,_ = self.decoder(in_s, img_embeddings)
                    pred_s = pred_s.view(-1, len(self.tokensizer))
                    out_s = out_s.view(-1)
                    loss = loss_fn(pred_s, out_s)
                    epoch_loss += loss.item()
                    loss.backward()
                    optimizer_decoder.step()
                    optimizer_encoder.step()

                    pbar.set_postfix(loss = f"{loss.item():0.4f}", refresh=True)
                    pbar.update(1)

                schedular_decoder.step()
                schedular_encoder.step()

            # if epoch%self.config.eval_steps == 0:
            print(f'epoch : {epoch}  train_loss: {epoch_loss/len(self.dataloader)}')
            
            self.encoder.eval()
            self.decoder.eval()
            val_loss = 0
            for img, in_s, out_s in tqdm(self.val_dataloader):
                with torch.no_grad():
                    img, in_s, out_s = img.to(self.device), in_s.to(self.device), out_s.to(self.device)
                    img_embeddings = self.encoder(img)
                    # img_embeddings = torch.unsqueeze(img_embeddings, dim= 0).repeat(self.n_lstm_layers, 1, 1)
                    pred_s,_ = self.decoder(in_s, img_embeddings)
                    pred_s = pred_s.view(-1, len(self.tokensizer))
                    out_s = out_s.view(-1)
                    loss = loss_fn(pred_s, out_s)
                    val_loss+= loss.item()

            print(f'epoch : {epoch}  val_loss: {val_loss/len(self.val_dataloader)}')

            torch.save(self.decoder.state_dict(), self.config.decoder['save_path'])
            torch.save(self.encoder.state_dict(), self.config.encoder['save_path'])
            self.encoder.train()
            self.decoder.train()


    def predict(self, img, max_length=40):
        """
        Generate a caption for a given image.

        Args:
            img (torch.Tensor): The input image tensor.
            max_length (int): The maximum length of the generated caption.

        Returns:
            str: The generated caption.
        """
        self.encoder.eval()
        self.decoder.eval()
        curr_tok = self.tokensizer.bos_token_id
        # curr_tok = 1024
        sequence = [curr_tok]
        # curr_tok = self.tokensizer.encode(curr_text, return_tensors='pt')
        h = c = None
        with torch.no_grad():
            img_embeddings = self.encoder(img.to(self.device))
            # h = c = torch.unsqueeze(img_embeddings, dim= 0).repeat(self.n_lstm_layers, 1, 1)
        for _ in range(max_length):
            with torch.no_grad():
                input_= torch.Tensor([curr_tok]).to(device=self.device, dtype=torch.int).unsqueeze(dim=0)
                if h is None and c is None:
                    logits, (h, c) = self.decoder(input_, img_embeddings)
                else:   
                    logits, (h, c) = self.decoder(input_, img_embeddings, h, c)
                logits = torch.softmax(logits[:, -1, :], dim=-1)
                logits  = torch.squeeze(logits)
                curr_tok = torch.argmax(logits, dim=-1).item()
            
            # curr_tok = self.tokensizer.decode(token_num)
            sequence.append(curr_tok)
            if curr_tok == self.tokensizer.eos_token_id:
                return self.tokensizer.decode(sequence)
            
        return self.tokensizer.decode(sequence)

    def predict_top_k(self,img, max_length=40, top_k=2):
        self.encoder.eval()
        self.decoder.eval()
        curr_tok1 = self.tokensizer.bos_token_id
        curr_tok2 = self.tokensizer.bos_token_id
        # curr_tok = 1024
        sequence1 = [curr_tok1]
        sequence2 = [curr_tok1]
        seq1_prob = [1]
        seq2_prob = [1]
        # curr_tok = self.tokensizer.encode(curr_text, return_tensors='pt')
        h = c = None
        with torch.no_grad():
            img_embeddings = self.encoder(img.to(self.device))

        for _ in range(max_length):
            with torch.no_grad():
                input_= torch.Tensor([[curr_tok1],[curr_tok2]]).to(device=self.device, dtype=torch.int)
                if h is None and c is None:
                    logits, (h, c) = self.decoder(input_, img_embeddings.repeat(2,1,1))
                else:   
                    logits, (h, c) = self.decoder(input_, img_embeddings.repeat(2,1,1), h, c)
                logits = torch.softmax(logits[:, -1, :], dim=-1)
                logits  = torch.squeeze(logits)
                probs, indices = torch.topk(logits, k= 2, dim=-1)
                
                curr_tok1 = indices[0]
                curr_tok2 = indices[1]
            
            # curr_tok = self.tokensizer.decode(token_num)
            sequence1.append(curr_tok1)
            sequence2.append(curr_tok2)
            if curr_tok1 == self.tokensizer.eos_token_id and curr_tok2 == self.tokensizer.eos_token_id:
                if probs[0] > probs[1]:
                    return self.tokensizer.decode(sequence1)
                return self.tokensizer.decode(sequence2)
            elif curr_tok1 == self.tokensizer.eos_token_id:
                return self.tokensizer.decode(sequence1)
            elif curr_tok2 == self.tokensizer.eos_token_id:
                return self.tokensizer.decode(sequence2)





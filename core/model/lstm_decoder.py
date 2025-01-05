import torch

class LSTMDecoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embd_size = config.decoder.embed_size
        self.hidden_size = config.decoder.img_embed_dim
        self.n_layers = config.decoder.n_layers
        self.vocab_size = config.vocab_size

        self.embeding_layer = torch.nn.Embedding(num_embeddings=self.vocab_size,
                                                 embedding_dim= self.embd_size)
        self.lstm = torch.nn.LSTM(input_size= self.embd_size, 
                                  hidden_size= self.hidden_size,
                                  num_layers = self.n_layers)

        self.project_layer = torch.nn.Linear(in_features= self.hidden_size, 
                                             out_features= self.vocab_size)
        
        self.softmax = torch.nn.Softmax(dim=-1)


    
    def forward(self, input):

        embeddings = self.embeding_layer(input)
        lstm_output = self.lstm(embeddings)
        projection = self.project_layer(lstm_output)
        logits = self.softmax(projection)

        return logits
        

class LSTMDecoderWithProjectLayer(torch.nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embd_size = config.decoder['embed_size']
        self.hidden_size = config.decoder['hidden_size']
        self.n_layers = config.decoder['n_layers']
        self.vocab_size = vocab_size
        self.projection_dims = config.decoder['projection_dim']
        self.lstm_input_size = config.decoder['lstm_input_size']

        self.embeding_layer = torch.nn.Embedding(num_embeddings=self.vocab_size,
                                                 embedding_dim= self.embd_size)
        self.lstm = torch.nn.LSTM(input_size= self.lstm_input_size, 
                                  hidden_size= self.hidden_size,
                                  num_layers = self.n_layers,batch_first=True)
        self.projection_head = torch.nn.Linear(in_features= self.hidden_size,
                                               out_features= self.projection_dims, bias=False)
        
        self.layer_norm = torch.nn.LayerNorm(normalized_shape= self.projection_dims)
        self.dropout = torch.nn.Dropout(0.5)
        self.activation = torch.nn.ReLU()
        
        self.output_layer = torch.nn.Linear(in_features= self.projection_dims, 
                                             out_features= self.vocab_size)
        
        self.softmax = torch.nn.Softmax(dim=-1)


    
    def forward(self, input, img_embeddings, h_n=None, c_n=None):
        # if h_n is None:
        #     h_n = torch.zeros(self.n_layers, img_embeddings.shape[0], self.hidden_size).to(input.device)
        # if c_n is None:
        #     c_n = torch.zeros(self.n_layers, img_embeddings.shape[0], self.hidden_size).to(input.device)

        embeddings = self.embeding_layer(input)
        # h_n = torch.unsqueeze(h_n, dim=0).repeat(self.n_layers, 1, 1)
        # c_n = torch.unsqueeze(c_n, dim= 0).repeat(self.n_layers, 1, 1)
        if len(list(embeddings.shape)) == 2:
            # embeddings = torch.unsqueeze(embeddings, dim=0)
            img_embeddings = img_embeddings.unsqueeze(dim=1).repeat(1, embeddings.shape[1], 1)
        else:
            img_embeddings = img_embeddings.repeat(1, embeddings.shape[1],1)
        embeddings = torch.cat([embeddings, img_embeddings], dim=-1)
        if h_n is not None and c_n is not None:
            lstm_output = self.lstm(embeddings, (h_n, c_n))
        else:
            lstm_output = self.lstm(embeddings)
        projection = self.dropout(self.activation(self.layer_norm(self.projection_head(lstm_output[0]))))
        logits = self.output_layer(projection)

        return logits, lstm_output[1]

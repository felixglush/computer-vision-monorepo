import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        # define model layers
        # for generating word embeddedings from the captions
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # for both the word embeddeings and the feature vectors from the Encoder
        self.lstm = nn.LSTM(input_size=embed_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        # generates predicted words
        self.output_vocab = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # pass the captions through the word embedding layer dropping the last <end> token so it doesn't predict a word after it
        # captions: (batch_size x caption length)
        # caption_embeddedings: (batch_size x caption length x embed_size)
        caption_embeddings = self.embedding(captions[:,:-1])
        
        # concatenate the image feature vectors with caption_embeddings to pass them into the LSTM network
        # features: (batch_size x embed_size)
        # Create a second dimension for features so we can concatenate along it
        lstm_inputs = torch.cat((features.unsqueeze(dim=1), caption_embeddings), dim=1)
        # lstm_output: (batch_size x caption_length x hidden_size) as per documentation and batch_first=True
        lstm_output, _ = self.lstm(lstm_inputs)
        
        # finally, pass through the fully connected layer to generate probabilities for the words in vocab
        # outputs: (batch_size x caption length x vocab size) as per notebook 2 step 4.
        # Why don't we need to flatten lstm_output to be accepted by the linear layer which has vocab_size number of input nodes?
        outputs = self.output_vocab(lstm_output)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        generated_caption_indices = []
        
        for word_i in range(max_len):
            output, states = self.lstm(inputs, states)
            
            possible_words = self.output_vocab(output.squeeze(dim=1))
            
            most_likely_word = possible_words.argmax(dim=1) # argmax is index of maximum value
            
            generated_caption_indices.append(most_likely_word.item())
            
            if most_likely_word.item() == 1: # <end> token
                break
            
            # make most_likely_word the input to the lstm for the next step word_i+1
            # remember to embed it to generate a semantic vector for the word
            inputs = self.embedding(most_likely_word)
            # because the lstm accepts 3 dimensional input but embedding returns 2 dims, add in a dimension
            # 1 x embed_dim -> 1 x 1 x embed_dim
            inputs = inputs.unsqueeze(1) 
        
        return generated_caption_indices
            
            
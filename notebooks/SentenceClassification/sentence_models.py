
import torch.nn as nn
import torch

import torch.nn as nn

class RNNTextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_classes):
        """
        RNNTextClassifier class constructor.

        Args:
        - input_size (int): The number of expected features in the input.
        - hidden_size (int): The number of features in the hidden state.
        - output_size (int): The number of features in the output.
        - num_classes (int): The number of classes in the classification task.
        """
        super(RNNTextClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, output_size)
        
        # Classifier layer for text classification
        self.fc = nn.Linear(output_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_sequence):
        """
        Forward pass of the RNNTextClassifier.

        Args:
        - input_sequence (torch.Tensor): The input sequence of shape (seq_len, batch_size, input_size).

        Returns:
        - final_output (torch.Tensor): The final output of the classifier layer of shape (batch_size, num_classes).
        """
        # Initialize hidden state
        hidden = self.rnn.init_hidden()
        
        # List to store the output at each time step
        outputs = []

        # Iterate through the input sequence
        for input_tensor in input_sequence:
            output, hidden = self.rnn(input_tensor, hidden)
            outputs.append(output)

        # Apply the classifier layer to the final output
        final_output = self.fc(outputs[-1])
        final_output = self.softmax(final_output)
        
        return final_output

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN module.

        Args:
        - input_size (int): The size of the input tensor.
        - hidden_size (int): The size of the hidden tensor.
        - output_size (int): The size of the output tensor.
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input_tensor, hidden_tensor):
        """
        Performs a forward pass through the RNN module.

        Args:
        - input_tensor (torch.Tensor): The input tensor.
        - hidden_tensor (torch.Tensor): The hidden tensor.

        Returns:
        - output (torch.Tensor): The output tensor.
        - hidden (torch.Tensor): The hidden tensor.
        """
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        """
        Initializes the hidden tensor.

        Returns:
        - torch.Tensor: The hidden tensor.
        """
        return torch.zeros(1, self.hidden_size)

class LSTM(nn.Module):
    """
    A class representing an LSTM model.

    Attributes:
    - hidden_size (int): The size of the hidden tensor.
    - num_layers (int): The number of LSTM layers.
    - embedding (nn.Embedding): An embedding layer to convert words into dense vectors.
    - lstm (nn.LSTM): An LSTM layer.
    - fc (nn.Linear): A linear layer for classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer to convert words into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        Performs a forward pass through the LSTM module.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - out (torch.Tensor): The output tensor.
        """
        # Input x should be of shape (batch_size, sequence_length)
        embedded = self.embedding(x)  # Embedding the input

        output, (hidden, cell) = self.lstm(embedded)
        # output dim: [sentence length, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]
        
        output = self.fc(hidden)
        return output

        # # Initialize hidden state with zeros
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # # Forward pass through LSTM
        # out, _ = self.lstm(embedded, (h0, c0))
        
        # # Get the output of the last time step
        # out = out[:, -1, :]
        
        # # Fully connected layer
        # out = self.fc(out)
        # return out

class GRU(nn.Module):
    """
    A class representing a GRU model.

    Attributes:
    - hidden_size (int): The size of the hidden tensor.
    - num_layers (int): The number of GRU layers.
    - embedding (nn.Embedding): An embedding layer to convert words into dense vectors.
    - gru (nn.GRU): A GRU layer.
    - fc (nn.Linear): A linear layer for classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer to convert words into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        Performs a forward pass through the GRU module.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - out (torch.Tensor): The output tensor.
        """
        # Input x should be of shape (batch_size, sequence_length)
        embedded = self.embedding(x)  # Embedding the input
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through GRU
        out, _ = self.gru(embedded, h0)
        
        # Get the output of the last time step
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        return out

import math


class Transformer(nn.Module):
    """
    A class representing a Transformer model.

    Attributes:
    - src_mask (torch.Tensor): A tensor representing the source mask.
    - pos_encoder (nn.Sequential): A sequential layer to encode the position of the input tensor.
    - transformer_encoder (nn.TransformerEncoder): A transformer encoder layer.
    - encoder (nn.Embedding): An embedding layer to convert words into dense vectors.
    - ninp (int): The size of the input tensor.
    - classifier (nn.Linear): A linear layer for classification.
    """
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, num_classes, dropout=0.5):
        """
        Initializes the Transformer model.

        Args:
        - ntoken (int): The size of the vocabulary.
        - ninp (int): The size of the input tensor.
        - nhead (int): The number of attention heads.
        - nhid (int): The size of the feedforward layer.
        - nlayers (int): The number of layers.
        - num_classes (int): The number of classes for classification.
        - dropout (float): The dropout rate.
        """
        super(Transformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhead))
        encoder_layers = TransformerEncoderLayer(nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        
        # Added classifier layer for text classification
        self.classifier = nn.Linear(ninp, num_classes)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        """
        Generates a square mask for the sequence.

        Args:
        - sz (int): The size of the mask.

        Returns:
        - mask (torch.Tensor): The mask tensor.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        """
        Initializes the weights of the model.
        """
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Performs a forward pass through the Transformer module.

        Args:
        - src (torch.Tensor): The input tensor.

        Returns:
        - out (torch.Tensor): The output tensor.
        """
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        # Apply the classifier layer
        output = self.classifier(output[-1, :, :])  # You can use all output layers for classification if needed
        return output

from transformers import BertModel

class BertText(nn.Module):
    """
    BERT-based text classifier.

    Args:
        num_classes (int): Number of classes for classification.

    Attributes:
        bert (BertModel): Pretrained BERT model.
        dropout (nn.Dropout): Dropout layer.
        fc (nn.Linear): Fully connected layer for classification.

    """
    def __init__(self, num_classes):
        super(BertTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the BERT-based text classifier.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, sequence_length).

        Returns:
            logits (torch.Tensor): Output tensor of shape (batch_size, num_classes).

        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

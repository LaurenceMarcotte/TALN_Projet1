"""
Code created by Laurence Marcotte

inspired by https://github.com/Jackthebighead/duplicate-question-pair-identification/blob/master/model2/siamese_bilstm.ipynb
and by the code from the LSTM implementation done during the course IFT6135

This is the implementation of BiLSTM-embedding as explained in the report.
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
            self,
            vocabulary_size=40479,
            embedding_size=300,
            hidden_size=512,
            num_layers=1,
            pad_index=0,
            num_class=2,
            num_add_feature=1,
            dropout_prob=0.2,
            learn_embeddings=False,
            _embedding_weight=None,
            similarity_task=False
    ):

        super(LSTM, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learn_embeddings = learn_embeddings
        self.similarity_task = similarity_task
        # nombre de passage dans le lstm (2 si similarity task)
        self.num_lstm = 2 if self.similarity_task else 1
        self.direction = 2  # si bilstm il y a 2 direction sinon 1
        self.num_class = num_class  # nombre de classes
        self.num_add_feature = num_add_feature  # si on ajoute des features en plus des phrases quand on fait la classification
        self.dropout_prob = dropout_prob  # la probabilité de dropout
        self.cos_sim = nn.CosineSimilarity(dim=1)

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=pad_index, _weight=_embedding_weight
        )
        self.lstm_a = nn.LSTM(
            embedding_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size * (self.num_lstm + self.num_add_feature) * self.direction +
                           2*int(self.similarity_task)),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(hidden_size * (self.num_lstm + self.num_add_feature) * self.direction +
                      2*int(self.similarity_task), self.num_class, bias=False),
        )

        # Freeze the embedding weights, depending on learn_embeddings
        self.embedding.requires_grad_(learn_embeddings)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs1, hidden_states1, inputs2=None, hidden_states2=None, feature=None):
        """LSTM.

        This is a Long Short-Term Memory network for classification. This
        module returns for each example the probability for belonging in a class.

        Parameters
        ----------
        inputs1 (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        hidden_states1 (`tuple` of size 2)
            The (initial) hidden state. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)

        inputs2 (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the second token sequences. This is only necessary if
            self.similarity_task is True.

        hidden_states2 (`tuple` of size 2)
            The (initial) hidden state for the second sequence. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            This is only necessary if self.similarity_task is True.

        feature (`torch.FloatTensor` of shape `(batch_size, 1)`)
            The feature added to self.similarity_task which is the number of common words between the sequences.
            This is only necessary if self.similarity_task is True.


        Returns
        -------
        log_probas (`torch.FloatTensor` of shape `(batch_size, num_class)`)
            A tensor containing the log-probabilities for each class

        hidden_states (`tuple` of size 2)
            The final hidden state. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """
        embedded_input = self.embedding(inputs1)
        lstm_output, (h, c) = self.lstm_a(embedded_input, hidden_states1)
        lstm_output = torch.sum(lstm_output, dim=1)

        if self.similarity_task:
            embedded_input2 = self.embedding(inputs2)
            lstm_output2, (h2, c2) = self.lstm_a(embedded_input2, hidden_states2)
            sent2 = torch.sum(lstm_output2, dim=1)
            lstm_output = torch.cat((lstm_output, sent2, lstm_output*sent2,
                                     self.cos_sim(lstm_output, sent2).unsqueeze(-1), feature), 1)

        classifier_output = self.classifier(lstm_output)
        log_proba = self.softmax(classifier_output)
        return log_proba, (h, c)

    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (self.num_layers*self.direction, batch_size, self.hidden_size)

        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        c_0 = torch.zeros(shape, dtype=torch.float, device=device)

        if self.similarity_task:
            h_0_2 = torch.zeros(shape, dtype=torch.float, device=device)
            c_0_2 = torch.zeros(shape, dtype=torch.float, device=device)
            return (h_0, c_0), (h_0_2, c_0_2)

        return h_0, c_0
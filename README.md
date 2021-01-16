# Sentiment Classification

The repo provides a brief comparison of PyTorch and Keras when working with Recurrent Neural Networks for sentiment classification in the English language.

```
├── scripts/
│   ├── data.pcsv
│   ├── keras_nlp.py
│   └── pytorch_nlp.py
└── docs/
```

The data are Twitter disaster feeds and therefore a binary classification problem. The last RNN version I used (and uploaded here) are Gated Recurrent Units or GRUs. The implementation in Keras is very straightforward as you just need to substitute, e.g. the "GRU" model for an "LSTM" model in the sequential Keras model. There is slightly more complexity in PyTorch as you need to be aware of the matrix dimensions and what each step outputs sequentially. For example, an LSTM outputs not only a prediction for each timestep but also an additional hidden unit. This is easy to follow in the debugger though since PyTorch builds the model using dynamic computation graphs. However you will notice there is much more boilerplate in the PyTorch script which is not necessarily a bad thing imo as it forces the user to know the fundamental concepts of NLP.

<table>
  <tr>
    <td>Vanilla RNN</td>
     <td>LSTM</td>
     <td>GRU</td>
  </tr>
  <tr>
    <td valign="top"><img src = "/docs/rnn.png" width="300", height="250"></td>
    <td valign="top"><img src = "/docs/lstm.png"></td>
    <td valign="top"><img src = "/docs/gru.png"></td>
  </tr>
 </table>


# Sentiment Classification

Brief comparison of PyTorch and Keras when working with Recurrent Neural Networks for sentiment classification in the English language.

```
├── scripts/
│   ├── data.csv
│   ├── keras_nlp.py
│   └── pytorch_nlp.py
└── docs/
```

The data are Twitter disaster feeds and therefore a binary classification problem. The last RNN version which was used (and uploaded here) are Gated Recurrent Units or GRUs.

<table>
  <tr>
    <td>Vanilla RNN</td>
     <td>LSTM</td>
     <td>GRU</td>
  </tr>
  <tr>
    <td valign="top"><img src = "/docs/rnn.png"></td>
    <td valign="top"><img src = "/docs/lstm.png"></td>
    <td valign="top"><img src = "/docs/gru.png"></td>
  </tr>
 </table>


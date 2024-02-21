# Building My Own...

## Graph Neural Network

There are more breakthroughs in AI (Software 2.0) which are interesting. Deep mind has made a very compute efficient weather forecasting model [recently](https://www.technologyreview.com/2023/11/14/1083366/google-deepminds-weather-ai-can-forecast-extreme-weather-quicker-and-more-accurately/#:~:text=Google%20DeepMind%E2%80%99s%20weather%20AI%20can,days%20sooner%20than%20traditional%20methods). Compute efficiency seems to be an important component of building powerful AI systems. [Quote from Ilya Sutskever](https://youtu.be/Ft0gTO2K85A?si=Mm9B2haYvS1FCoAH&t=1653). Also 
the climate is a dynamic complex system. The increasing success of predicting the weather with neural networks is an indication that NNs can model dynamically complex systems. This is a deep lesson, since it could extend to other dynamically complex systems.
 Also 
the climate is a dynamic complex system. The increasing success of predicting the weather with neural networks is an indication that NNs can model dynamically complex systems. This is a deep lesson, since it could extend to other dynamically complex systems.

 
Graph neural networks receive graphs as input and output graphs as output. A graph includes the nodes, edges and global information about it.

There are a couple ways to represent the graphs. In the pytorch geometric package here are how graphs are represented:

`Graph(x, edge_index, y, train_mask, val_mask, test_mask)`

**x**: a tensor representing the nodes in the graph and the tensor
**edge_index**: 2 vectors of the same size. The first has the initial node, the second the destination node (in the matching index)
**y**: labels of the data
**train_mask**: a vector of the size the data, but it has True and False values which indicate which data is meant for training
**val_mask**: same as train_mask but for validation set
**test_mask**: same as val_mask but for test set

>**A GNN is an optimizable transformation on all attributes of the graph (nodes, edges, global-context) that preserves graph symmetries (permutation invariances).**

 The most simple GNN is just adding a linear layer after every component of the graph. This is doesn't really take advantage of relationships represented in the graph, but it's the simplest model.
 In this model, to make the predictions after the last layer we can do a pooling technique, which basically is aggregating the embeddings of the neighboring nodes using some aggregation function like avg, sum, etc, then feeding into the a linear layer.

### Simplest Version

 The most simple GNN is just adding a linear layer after every component of the graph. This is doesn't really take advantage of relationships represented in the graph, but it's the simplest model.
 In this model, to make the predictions after the last layer we can do a pooling technique, which basically is aggregating the embeddings of the neighboring nodes using some aggregation function like avg, sum, etc, then feeding into the a linear layer.

[Link to implementation](Graph_Neural_Network/GNNs.ipynb)

### With Message Passing

In this implementation, the graph connectivity is not used at all inside the GNNs layers and only at the time of prediction. We can improve this by implementing pooling mid GNN layers, which is also called message passing.

[Link to implementation](Graph_Neural_Network/GNNs_Message_Passing.ipynb)


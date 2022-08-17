# End to End Question Answering system
## Intro
For this project, we implement a basic question answering system. Namely, the system takes as input a question in natural language and identify the answer in a body of text(s). The system should be built to take input as provided by the dataset given below, which also provide additional details. 

Inference of the model is introduced as a service, which exposes an http endpoint where a client can post a question and a context expecting to get a relative answer. 

## Project Structure
The design of the complete system is structure in a way to be easily maintained and deployed. For this reason is decomposed into different components serving various purposes. A good opinionated practice which we adapt to our design is referenced 

## Model definition
The model is an adaption from the  [Document Retriever](https://arxiv.org/pdf/1704.00051.pdf) the first model having a great impact in advancing neural question answering approaches. The reason behind adapting this model is that it provides a motive for investigating and designing all those components, that solidify neural translation model models, before the most recent trends e.g transformers. 

The model requires as inputs the context and the question which are vectorized through an extensive preprocessing stage.
The output of the model is the probabilities of the start/end index of the answer with respect to the questin inside the context e.g the span of the answer.

## NLP and Deep Learning frameworks

## Integrating model as a service
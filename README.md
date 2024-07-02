Ragam is a term used in Indian classical music to describe the melodic framework for a musical 
composition. Each ragam is characterized by its own distinct musical notes and scales, and is associated 
with a specific mood or emotion. Identifying the ragam of a given musical phrase can be a difficult task for 
beginners, as it requires a deep understanding of the musical theory and structure of each ragam.
This project aims to address this issue by utilizing machine learning techniques to predict the ragam of a 
given musical phrase. The project will train a model on a dataset of phrases with their corresponding ragams, 
and use the trained model to predict the ragam of new phrases. This will provide beginners with an easyto-use tool for identifying the ragam of a given phrase, allowing them to better understand and appreciate 
the nuances of Indian classical music.
The neural network model created for this assignment is a sequential model with an Embedding layer, an 
LSTM layer, and a Dense layer. the Dense layer, which uses a softmax activation function, generates a 
probability distribution across the set of potential ragas, allowing the model to predict the most likely raga 
for each given input sequence.
The model is trained utilising the categorical cross-entropy loss function and the Adam optimizer, with 
accuracy serving as the performance metric. Training lasts 10 epochs, with a validation split to assess the 
model's performance on previously unknown data. The model performs satisfactorily, demonstrating its 
capacity to learn the mapping between musical phrases and ragas efficiently. Following training, the model's
accuracy is assessed over the full dataset to demonstrate its resilience and generalizability. A single input 
sequence can be tokenized, padded, and fed into the trained model to predict the raga. The prediction 
technique is translating the sequence into the same format as the training data and applying the model to 
create a probability distribution over all possible ragas, from which from raga is selected.

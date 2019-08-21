**recurrent neural networks**

chapter 4 exercises
-------------------

4.1
---
Assume that our corpus starts out " *STOP* I like my cat and my cat 
likes me. *STOP*" . Also assume that we assign individual words their unique
integer as we read in the corpus, starting with 0. If we have a batch size 5,
write out the values we should read in to fill the placeholders `inpu` and `answr`
on the first training batch.

ans:

batchSz = 5 
window = 2
numBatches = 2

*STOP*  |  I 
like    |  my 
cat     |  and
my      |  cat
likes   |  me
*STOP*  |
```python
inpu = [  [1, 0, 0, 0, 0, 0, 0, 0, 0],  # *STOP*
          [0, 0, 1, 0, 0, 0, 0, 0, 0],  # like 
          [0, 0, 0, 0, 1, 0, 0, 0, 0],  # cat     
          [0, 0, 0, 1, 0, 0, 0, 0, 0],  # my
          [0, 0, 0, 0, 0, 0, 1, 0, 0] ] # likes

answr = [ [0, 1, 0, 0, 0, 0, 0, 0, 0],  # I   
          [0, 0, 0, 1, 0, 0, 0, 0, 0],  # my
          [0, 0, 0, 0, 0, 1, 0, 0, 0],  # and
          [0, 0, 0, 0, 1, 0, 0, 0, 0],  # cat
          [0, 0, 0, 1, 0, 0, 0, 1, 0] ] # me
```


4.2
---
Explain why, if you hope to have any chance of learning a good embedding - based language
model, you may not set all of **E** to zero. Make sure your explanation also works for setting all of
**E** to one.

ans: if you start off with all of **E** as zero your model will never get better. Because there will
be no way to choose the first few predictions. The model would have to make an arbitrary choice
from all equal probabilities. Setting all of **E** to one also is the same case because now all the
words are equally likely and the model will have to arbitrarily choose  from all of them. The model isn't
actually attempting to predict because all of them start off the same.

4.3
---
Explain why, if you are using L2 regularization, it is positively a bad idea to compute the actual
total loss.

ans: Actually computing the total loss entails adding the sum of squares of all the weights. The tensorflow
computation graph does not actually need this value anywhere else, so it would slow down the training
to compute it. For example, to computing the derivative of the loss with respect to w_ij
only requires adding w_ij to the total loss. 

4.4
---
Consider building a trigram - fully - connected language model. In our version
we concatenated the embeddings for the two previous inputs to form the model input. Does the order
in which we concatenate have any effect on the model's ability to learn ? Explain.

ans: It depends on how the model is set up. Technically no. If the model treats the two previous words
with the same importance, then it should have no difference. This is in the case where batches don't overlap.
If the batches are overlapping (ie window size is 3, but each batch only shifts 1 or 2 right), this means
the order of concatenation does matter because a single word is tied to not just one word, but two; its 
tied to a word in reference as the immediate previous and another word in reference as the second previous

4.5
---
Consider an nn unigram model. Can its model perplexity be any better than picking words
from a uniform distribution ? Why or why not ? Explain what pieces of the bigram model are needed
for optimal performance of a unigram model.

ans: Yes, its model perplexity would be better than picking words from a uniform distribution because
even though there is no input, more common words would be guessed (ie the). There is no point of reference. 
There is no input. The general frequencies of words could be kept track of through biases. 

4.6
---
A *linear gated unit* (LGU) is a variant of LSTMS. Referring back to Figure 4.9, we see that the latter
has one hidden layer that controls what gets removed from the main memory line, and a second that
controls what is added. In both cases the layers take the lower line of control as input, and 
produce a vector of numbers between 0 and 1 that are multiplied with the memory line (forgetting) or 
added to it (remembering). LGUs differ in replacing these two layers by a single layer with the same input.
The output is multiplied by the control line as before. However, it is also subtracted from one, multiplied
by the control layer, and added to the memory line. In general, LGUs work as well as LSTMs and, having one 
fewer linear layer,
 are slightly faster. Explain the intuition. Modify Figure 4.9 so it represents the workings of a LGU.
 
 ans: I would say the intuition has to do with the fact that all the information is stored in this one
 memory pipeline instead of the two in lstms. In lstms, there seems to be some redundancy where similar
 transformations are applied more than once because of the need to update the memory line and the regular
 h - line. The GRU would solve this with the single pipeline. Also between the memory line or cell state
 and current input *h*, there might also be some redundancy because some information might be stored. For example,
 say you were currently processing a noun phrase and you wanted the neural network to keep track of the 
 type of noun (singular or plural). The h line might store this information and the cell state would also
 store this information because it might be relevant to the task.

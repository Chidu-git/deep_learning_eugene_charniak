**sequence to sequence learning**

chapter 5 exercises
-------------------

5.1
---
Suppose we are using a multiple - length seq2seq for an  MT 
program and have deciced on two sentence sizes, one for up to 7
words (and STOPs) for English and 10 for French and the other
for up to 10 words for English and 13 for French. Write out the
input if the French sentence is "A B C D E F" and the English is
"M N O P Q R S T".

ans: smaller sentence size -> french = 'STOP A B C D E F STOP STOP STOP',
                              english = 'STOP M N O P Q STOP'
                              
     larger sentence size -> french = 'STOP A B C D E F STOP STOP STOP STOP STOP STOP',
                              english = 'STOP M N O P Q R S T STOP'    

5.2
---
We chose to illustrate attention in section 5.3 with a particularly
simple form, one that based the attention decision only on the location
of the attention in both the French and English. A more sophisticated
version bases the decision on the input state vectors to the English
position we are working on and the proposed state vector whose
influence we are deciding. While this can allow more sophisticated
judgements, it requires a significant complication of the model. In
particular, we can no longer use standard tf recurrent network back
propagation through time for the decoder. Explain why.

ans: This wouldn't work because in our current simple model 
because the second rnn would then have the regular propagation
, left to right, translating target words one at a time, but it 
also has to factor in the effect of the proposed state vector. Factoring
in the actual proposed state vector means that we have to look at all
the state vectors at the same time. We don't do this anywhere else 
and the graph doesn't have the capability to look forward. The intuitive 
explanation would say that the rnn is going left to right, but we also 
added the influence of the state vectors. How do I know how this state
vector will effect the rnn unit to the right at time (t + 1) when I'm
still at time (t) ? 

5.3
---
It has frequently been observed that feeding the source language
into the seq2seq encoder *backward* (but leaving the decoder working
forward) improves MT performance by a slight but constant amount. Make
up a plausible story for why this would be the case.

ans: This might be the case in the singular pipeline feed forward
seq2seq model because the last word from the source language is directly connected
to the first word of the translation prediction. This might inadvertently
give the last word more weight to the overall translation and especially
the beginning of the translation. By feeding in the source language
into the encoder language backwards, this means the last word encoded (ie 
first word in source language) is
directly connected to the first word of the translation prediction.
By intuition this is probably better than the first word. 

5.4
---
In principle, we could have a seq2seq model with two losses that
we add together to make the total loss. One would be the current
MT loss incurred by not predicting the next target word with 
probability 1. The second could be a loss in the encoder, asking
the encoder to predict the next source (e.g., French) word -- i.e.,
a language model loss. (a) Make up a plausible story for why this 
will degrade performance. (b) Make up a plausible story for why
this will improve performance.

ans: (a) A plausible story for why this might degrade performance
is the fact that essentially two models with different goals are
being trained. This could put the losses at odds with each. Which
would only degrade performance in reference to the the main model of
predicting the next target word. It's unnecessarily complicating the model
more when predicting your encoding is not necessary.

(b) A plausible story for why this might improve performance would 
be that the goal of this model is to map this source language to
the target language. if the representation for the source language
is flawed, it makes it technically impossible to find the 
exact mapping because the input is wrong. if the source language 
model has less error it might make the mapping to the target
language closer to the exact solution. Practically the most direct
improvement from this dual loss model would be the performance of the
model on new French words because a better structure for French has
been encoded. 

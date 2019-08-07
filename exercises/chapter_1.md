chapter 1 written exercises
---------------------------
1.1
---
Consider our feed - forward Mnist program with a batch size of one. Suppose we look at the bias
variables before and after training on the first. example. If htey are being set correctly (i.e. 
if there are no bugs in our program) describe the changes you should see in their values.

ans: If the bias (b_i) corresponds to that specific first image (a) then the b_a would increase.
The rest of the biases should decrease because they're not the right digit

1.2
---
We simplify our Mnist computation by assuming our "image" has two binary - valued pixels, 0 and 1,
there are no bias parameters, and we are performing a binary classification problem. (a) Compute
the forward - pass logits and probabbilities when the pixl values are [0, 1] and the weights are:

     .2  -.3
    -.1   .4
    

Here w(i,j) is the weight on the connection between the ith pixel and the jth unit. E.g., 
w(0, 1) = - .3. (b) Assume the correct number is 1 (not 0) and use a learning rate = .1. What is 
the loss ? Also compare gradient w_0_0 on the backward pass.

ans:
```python
    A =[ (0)*(.2) + (1)*(- .1), (0) * (- .3) + (1) * (.4) ] = [ - .1, .4 ]
    sigmoid(A0) = (e^ (- .1) / [ e^(- .1) + e^(.4) ]) 
    sigmoid(A1) = (e^ (.4) / [ e^(- .1) + e^(.4) ])
    
    A = [ .3775, .6224 ] 

    L(A1) = - log(.6224) # a = j so Pr(A(x)) = 1 - p_j
    L(A0) = - log(.3375)  # a != j so Pr(A(x)) = - p_j
    
    gradient_w00 = w_0_0 * partialX / partial L_j 
                 = .2 * partialX / partial_L_0 = .2 * ( - .3375 ) 
                 = .2 * partialX / partial_L_1 = .2 * ( 1 - .6224)
```

1.3
---
Same question as 1.2 except the image is [0, 0].

ans:
```python
    A = [0, 0]
    sigmoid(A0) = .5
    sigmoid(A1) = .5
    
    L(A0) = - log(.5)
    L(A1) = - log(.5)
   
    gradient_w00 = w_0_0 * partialX / partial L_j 
                 = .2 * partialX / partial_L_0 = .2 * (- .5) 
                 = .2 * partialX / partial_L_1 = .2 * (- .5) 
```
1.4
---
A fellow student asks you, "in elementary calculus we found the minima of a function by
differentiating it, setting the resulting expresssion to zero, and then solving the equation. 
Since our loss function is differentiable, why don't we do that rather than bothering with 
gradient descent ?" Explain why this is not in fact possible.

ans: This isn't possible because even though the loss function is differentiable, it's too 
     computational expensive. This loss is a function, depending on the size of the data set
     and corresponding nodes and layer size, of thousands if not more variables. Partially
     differentiating with respect to all of them and using substitution to find the extreme
     minima is not possible.
     
1.5
---
Compute the following:
    
    [ 1 2 ] [ 0 1 ] 
    
    [ 3 4 ] [ 2 3 ]   +  [4 5] 
   
 ans: 
 ```python
[ [4, 7],
  [8, 15]]
  
  +
  
  [4, 5]
  [4, 5]
  =
  
  [[8, 12],
   [12, 20]]
``` 

1.6
---
In this chapter we limited ourselves to classification problems, for which cross entropy is typically 
the loss function of choice. There are also problems where we may want our nn to predict particular
values. For example, undoubtedly many folks would like a program that, given the price of a particular
stock today plus all sorts of other facts about the world, outputs the price of the stock tomorrow. 
If we were training a single - layer nn to do this we would typically use the squared - error loss:
    
    L(X, y) = (t - l(X, y))^2
    
where t is the actual price that was achieved on that day and l(X, y) is the output of one layer nn with
y = {b, W}. (This is also known as quadratic loss.) Derive this equation for the derivative of the loss
with respect to b_i.

ans: 
```python
    partial_deriv_b_i = 2 (t - l(X*W_i + b_i)) * (1)
```
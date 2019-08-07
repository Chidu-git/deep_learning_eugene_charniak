chapter 2 written exercises
---------------------------

2.1
--- 
 what would be the result if in figure 2.5 we had instead computed tf.reduce_sum(A), where 
         A is the array on the left of the figure ?
         
  In figure 2.5, the actual command that is called =>
         
 ```python
 tf.reduce_sum(A, reduction_index=[1])
 ```
 
 The difference between this and the following command
  ```python
 tf.reduce_sum(A)
 ```  
is that 
  ```python
 tf.reduce_sum(A, reduction_index=[0]) = tf.reduce_sum(A) = [0, 0, 3.2, 0, .9] 
 ```
 
 ans: The reduction indexes refer to how A is summed, [0] is default and over columns and [1] over
 rows
 
2.2 
---
What is wrong with taking line 14 from Figure 2.2 and inserting it between line 22 and line 23,
        so that the loop now looks like:
        
```python
for i in range(1000):
    imgs, anss = mnist.train.next_batch(batchSz)
    train = tf.train.GradientDescentOptimizer(0.5).minimize(xEnt)
    sess.run(train, feed_dict={img : imgs, ans : anss})

```

ans: You're essentially calling and/or redefining the gradient descent optimizer in each pass, which is 
     unnecessary. 'train' is being called by mnist and then being redefined in the line underneath.
     

2.3
---
Here is another variation of the same line of code. Is this ok ? If not, why not ?

```python
for i in range(1000):
    img, anss = mnist.test.next_bach(batchSz)
    sumAcc += sess.run(accuracy, feed_dict={img:img, ans:anss})

```

ans: You're essentially saying the whole batchSz just maps to one image. I don't know what the outcome 
     of this code would be
     
     
2.4
---
In figure 2.10, what would be the shape of the tensor output of the operation
```python
tensordot(wAT, encOut, [[0], [1]]) ?
Explain.

```

ans: wAT is a 4 x 3 matrix and encOut is a 2 x 4 x 4
     tensordot is doing dot product of the rows of wAT (i.e. 0) with the columns of encOut (i.e. 1)
     The resulting matrix would be 3 x 2 x 4
     
2.5
---
Show the computation that confirms that the first number in the tensor printed out at the bottom of 
the example in Figure 2.10 (.8) is correct (to three places)

ans: <1, 1, 1, -1> * <.6, .2, .1, .1>^T = .8 

2.6
---
Suppose the input has shape [50, 10]. How many TF variables are created by the following:
```python
    01 = layers.fully_connected(input, 20, tf.sigmoid) ?
```
What will the standard deviation for the variables in the matrix created ?

ans: (50 x 10) * (10 x 20) = 50 x 20
       
Xavier_initialization = sqrt(1 / (in + out))

X_initialization = sqrt(1 / 30) 

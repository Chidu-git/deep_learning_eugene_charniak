chapter 3 exercises
-------------------
3.1
---
(a) Design a 3 * 3 kernel that detects vertical lines in a lback and white image, and returns 
the value 8 when applied to the upper - left - hand side of the image in figure 3.2. It should 
return zero if all the pixels in a patch are of equal intensity. (b) Design another such kernel.

ans:
```python
-2  1  1 
-2  1  1 
-2  1  1 

-2  1.5  .5 
-2  1.5  .5 
-2  1.5  .5
```

3.2
---
In our discussion of equation 3.2 we said in an off - hand comment that the size of the convolution
filter had no impact on the number of application whens using same padding. Explain how this 
can be.

ans: The size of the convolutional filter has no impact because SAME padding means that
     the filter is still applied so long as there is at least one pixel on the extreme left or 
     extreme top. The ones that hang off on the right or the bottom are filled in with averages
     of the patches. Hence only the stride and size of the image impact  SAME padding
     
     
3.3
---
In our discussion of padding we said that Valid padding *always* yields an output image having
smaller 2d dimensions than the input. Strictly speaking, this is not the case. Explain the 
(relatively unininteresting) case when the statement is false.

ans: In the case of a square 2d image where the filter perfectly divides the image in patches
and the number of filters applied is image Dim / filter Dim, then there won't be any decrease
in size from input to output. Generally speaking valid padding reduces the input size because
the filter and stride don't cleanly divide the image into patches. This results in edges
being ignored because there isn't a complete filter to cover them.

3.4
---
Suppose an input to a convolution nn is a 32 * 32 color image. We want to apply eight convolution
filters to it, all with shape 5 * 5. We are using valid padding and a stride of two both vertically,
and horizontally. (a) What is the shape of the variable in which we store the filters' values ?
(b) What is the shape of the output of tf.nn.conv2d ?

ans:
```python
numPatchConvolutions = 14 = floor((image_h - filter_h + 1) / stride_h)
flts = [batchSz, 14, 14, 8]
output = 14 * 14 and 8 channels
```

3.5
---
Explain what the following code does differently from the almost identical code at the beginnning
of Section 3.4.2:
```python
    convOut = tf.nn.conv2d(image, flts, [1,1,1,1], "SAME")
    convOut = tf.nn.maxpool(convOut, [1,2,2,1], [1,1,1,1], "SAME"). 
```
In particular, for an arbitrary values of image and flts, does convOut have same shape, in both cases ?
Does it necessarily have the same values ? Is one set of values a proper subset of the other ? In each
case, why or why not ?

ans: Because convolutional filters reduce the size of an image, this can become extreme in the case of really
     large images. The use of maxpool allows you to first convolve with no reduction in size. Then maxpool
     goes through and takes the max value for a filter over a region of the image. There is also avg pool 
     which works the same way, but instead of the max over the region, it takes the average. The purpose of this
     is to try and mitigate the loss of data in an image through subsequent convolutional filters. The reduction
     becomes very extreme especially with a softmax function applied at the end of each layer. Both end up with 
     the same shape. It does not necessarily have the same values. The one without maxpool implemented at the 
     beginning of section 3.4.2 will always be a proper subset of the first line of code above where all the 
     strides = 1. The strides being one makes this the case being you're applying the filter the max number of times
     that is possible, so the original code with stride = 2 will be a proper subset of this. Its not the same 
     the other way around otherwises both sets would be equal.

3.6
---
(a) How many variables are created when we execute the following layers command ?
```python
    layers.conv2d(image, 10, [2, 4], 2, "SAME", use_bias=False) 
```
Assume the image has shape [100, 8, 8, 3]. Which of these shape values are irrelevant to the answer ? (b) How
many are if use_bias is set to True (the default) ?

ans: Each kernel we create has a shape of [2, 4, 3] which implies 24 variables per kernel. Since we 
create 10 of them, 240 variables are created. The first three dimensions of the image are irrelevant 
to the answer. If use_bias is set to true then the first dimension (100) is relevant because the biases have
to match up. 
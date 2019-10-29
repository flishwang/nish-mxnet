modified from git@github.com/apache/apache-mxnet/examples/image-classfication

# Hish: Better Activation Function with Both High-order Continuity and High-fidelity to Input Signals
Author: Bo Wang (flish_wang@sina.com)

I want to post this work to arxiv, but I don't have an arxiv account, can anyone help me?

# Introduction & Theorical Analysis
In deep neural netowrks, convolutional and fully-connected layers transform signals linearly, across different channels and locations, 
while activation functions transform signals non-linearly element-wisely. 
In the past years, many activation functions are proposed, such as tanh, sigmoid, leaky relu, softrelu, swish, mish and so on.

During these activation functions, relu is used in modern nerual networks most commonly. The well-acknowledgement reason that it defeats tanh and sigmoid functions is that the deriv of relu will not dimish when the signals falls away from zero.
However, as negative signals will generate zeros in the backward proportion, the weights in the convolutional or fully-connected layers may become dead. Therefore, softrelu, swish are proposed. Recently, mish, which are very similar to swish, shows consistently better performance than swish and
mish, But its difference and reason it defeating swish are not studied very well.

In this study, I analyzed the studies of mish, swish and relu, and make the following hypotheses that a good activation function must satisfy:

[H1.] The function should have continuous high-order derivative. SGD optimizer requires first-order derivite and ADAM optimizer may suppose the function
have second-order derivitive. Therefore, we need the continuity for at least two orders.
[H2.] The function curve should have a flat bottom and small non-positive derivative on the left of its minimal. This gives the network the ability to store non-active feature values around or on the left of the minimal. The non-positive derivative on left of the minimal makes the net mapping non-active feature values to a more concentrate range.
[H3.] The function should not have symmetric parts, as this will lose unecessary information during forward.
[H4.] The function curve should have line segment with derivitive of ONE on a range that there are enough input signals fall into this area. 

Hypothesis H4 is the most possible explaination that mish out-performs swish, as they have very similar values and first-order derivitives on the negative parts. The only obvious difference between mish and swish is that mish becomes identiy function earlier than swish as the input values raise, therefore, it transforms the postive signals with higher-fidelity than swish.

I conclude that the ability to transform enough signals with high-fidelity is necessary for a good activation function. This is because that the DNNs requires low-level features when making high-level features and the final predictions, which are generated in the earlier layers -- this also explains that why ResNet and densebox work: they provides earlier features directly to deeper layers. Activation functions like relu and mish transform positive signals with high-fidelity, then the activated features can be passed to where they are used directly after generated. That is why these functions have a better performance than their competitor tanh and swish.

Based on the four hypotheses, I designed a new series of activation function:
f(x) = 
	x	(if x >=0 )
	a_1 exp(x) + a_2 exp(2x) + a_3 exp(3x) + ...	(otherwise)
, which satisfies f(0)=0, f'(0)=1 (from both part) and f''(0) = 0 ( from both part).
It is easy to proof that these functions satisfies H1, H2, H3 and H4.

Among the function series, the simplest one is:
f(x) = 
	x	(if x >=0) 
	-2.5*exp(x) + 4*exp(2x) + -1.5*exp(3x)  (otherwise)
, which I named as Nish.

Compared with Mish, Nish becomes identiy function earlier, has a more simpler fomuler, thus is easier to compute both forwardly and backwardly. 

In this work, I applies mish and nish on cifar, minist and ImageNet in different networks (lenet and resnet) based on the mxnet official examples.
The experiments shows (.......)

# Experimental result
training scripts see [scripts.sh]

arguments:
cifar10: max-random-aspect-ratio 0.1 max-random-rotate-angle 15 max-random-shear-ratio 0.1 random-crop 1 pad 4 resize 32 lr 0.05
epoch 250 lr-step 150,200

| Act function                  | Dataset                                  | Network          | validation-accuracy
| ----------------------------- | ---------------------------------------- | ---------------- | -------------------
| relu                          | cifar10,3x32x32                          | res50            | 0.9505 (epoch 220)
| mish                          | cifar10,3x32x32                          | res50            | 0.9525 (epoch 192)
| nish                          | cifar10,3x32x32                          | res50            | 0.9537 (epoch 202)


# to be continued


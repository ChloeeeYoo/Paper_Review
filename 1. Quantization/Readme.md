# Paper List

## PACT
https://paperswithcode.com/paper/pact-parameterized-clipping-activation-for  
* Auxilixary classifier  
=> In some cases, it is possible to calculate the cross-entropy and cost function at intermediate layers of a neural network. This is done by using a technique called “auxiliary classifiers” or “auxiliary heads”. These auxiliary classifiers are added to intermediate layers of a neural network and are trained to predict the class labels. The cross-entropy and cost function can then be calculated at these intermediate layers as well.  
=> During the training process, you can calculate the auxiliary losses using the main targets. Here's the updated section of code for calculating the auxiliary losses.

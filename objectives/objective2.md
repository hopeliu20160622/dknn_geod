# GENERAL
- [x] Create new class of gDkNN for geodesic distances.
- [] Read carefully the experiments from the original DkNN https://arxiv.org/abs/1803.04765.
- [] Explore implementation of the original DkNN and experiments already implemented in the model zoo from cleverhans https://github.com/tensorflow/cleverhans/tree/master/cleverhans/model_zoo/deep_k_nearest_neighbors. This can save a lot of time.

# IMPLEMENT EFFICIENT GEODESIC KNN
- [] Include efficient implementation of gkNN on the NNGeod class, code available from https://mosco.github.io/geodesicknn/geodesic_knn.pdf.
- [] Continue institutionalization of the code.
- [] Might be a very good idea to train models for MNIST, SVHN, GTSRB and save them to share through git.

# GENERAL ACCURACY
To show potential benefits of the method, compare accuracies of softmax, DkNN and gDkNN (Table 1) on the following datasets:
- [] MNIST
- [] SVHN
- [] GTSRB
We might be able to do the same with datasets with noise or datasets for which we control flipped labels.

# ACCURACY VS CONFIDENCE AND CREDIBILITY
Explore if the "robustifying" effect is maintained for gDkNN (Figure 2). 
This Figure is particularly weird might be just for us to create 3x3 Figure just to see.

# ROBUSTNESS
Begin implementation of simple attacks from Berkeley https://arxiv.org/abs/1903.08333.
A way to show that the method "robustifies" a classifier is by showing necessary "lengths" of attacks.
The Figure could be a line comparing number of successfull attacks (y axis) vs the radius of ball (x axis).
- [] Mean attack, compare the "length" of the direction between DKNN and gDKNN the intuition is that the attacks have be more "perceptible" for gDKNN. We can compare the mean of the distribution of the lengths needed to change classes.
- [] Gradient based attack. We could see again the success of label flips compared the the radius of balls, in such way we can again show that gDkNN is more "robust".
- [x] Naive attack is meant for just one neighbor, which makes the comparison between gDkNN and DkNN futile.

# INTERPRETABILITY
- [] Maybe look qualitatively for mislabeled inputs in MNIST, and SVHN just like DKNN paper (Figure 3) of original DkNN. By looking for outliers in confidence.
- [] Maybe we can explore directly the flipping labels here to see if we are able to spot the flips.

# CONFIDENCE AND CREDIBILITY
- [] Try to replicate the distributions of softmaxconfidence, DkNNcredibility, gDkNNcredibility and normal  (Figure 3) of original DkNN.
    We can do this by rotating MNIST or including observations outside of MNIST directly, https://stackoverflow.com/questions/53273226/how-to-rotate-images-at-different-angles-randomly-in-tensorflow.

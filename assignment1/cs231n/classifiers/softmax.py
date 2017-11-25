import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores) # normalization trick, safe
        loss += -scores[y[i]] + np.log(np.sum(np.exp(scores)))
        for j in range(num_classes):
            dW[:, j] += np.exp(scores[j])/sum(np.exp(scores)) * X[i]
            dW[:, j] -= (j == y[i])* X[i] # Additional grad for j==y[i]
  # setting correct coefficients for data loss and reg loss
  loss /= num_train
  loss += 0.5*reg*np.sum(np.square(W))
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  scores = X.dot(W)
  scores -= np.max(scores, axis=1).reshape(-1, 1) # shifted
  exp_scores = np.exp(scores)
  sum_exp = np.sum(exp_scores, axis=1).reshape(-1, 1)
  log_sum_exp = np.log(sum_exp)
  fyi = -1 * scores[range(num_train), y].reshape(-1, 1)

  loss = np.sum(fyi + log_sum_exp)/num_train# data loss computed
  loss += 0.5 * reg * np.sum(np.square(W)) # plus reg loss
  
  temp = exp_scores/sum_exp
  temp[range(num_train), y] -= 1
  dW = np.transpose(X).dot(temp)
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


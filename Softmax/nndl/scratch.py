import numpy as np

score = np.dot(X, self.W.T) 
score -= np.max(score, axis=1, keepdims=True) # for numerical stability

# find softmax
exp_ = np.exp(score)
probabilities = exp_ / np.sum(exp_, axis=1, keepdims=True)

#loss = np.sum([np.log(pi) for pi in probabilities]) / size
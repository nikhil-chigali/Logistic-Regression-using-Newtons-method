
# coding: utf-8

# In[32]:


# Importing Dependencies
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[149]:


# Loading Data

Data = pd.read_csv('ClassificationData.txt', header = None)

# Info

Data.info()


# In[151]:


Data.head(10)


# In[152]:


# Preprocessing Data

X = Data.values[:, :2]
y = Data.values[:, 2].reshape((-1,1))

m, n = X.shape

# Adding Bias term

X = np.hstack(( np.ones(shape = (m,1)), X ))


# In[153]:


# Visualizing data
for i in range(m):
    if y[i] == 0:
        plt .scatter(X[i, 1], X[i, 2], marker = '_', color = 'r', s = 45)
    else:
        plt.scatter(X[i, 1], X[i, 2], marker = '+', color = 'b', s = 45)
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.show()


# In[155]:


# Initializing weights

"""
    X = m x n
    W = n x 1
    b = 1 x 1
    y = m x 1
"""
def rand_init_weights():
    np.random.seed(5)
    W = np.random.rand(n,1)    # Weights / Coefficients
    b = np.random.rand(1,1)    # Intercept term
    Theta = np.vstack((b, W))  # Joining W and b into one single vector for easy calculations
    return Theta


# In[156]:


# Defining sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[162]:


## Cost / Risk (Negative log likelihood)

"""
    Using Newton's method, we'll be maximizing the log likelihood of our predictor function
    In other words, we can say that we'll be minimizing the cost/loss/risk of our predictor function
"""

def cost(X, y, Theta):
    m = y.size
    
    z = X @ Theta
    a = sigmoid(z)
    
    J = -(1/m)*np.sum(y * np.log(a) + (1-y) * np.log(1-a))
    
    return J
  
    
## Gradient with respect to Theta

def gradient(X, y, Theta):
    m = y.size
    
    z = X @ Theta
    a = sigmoid(z)
        
    grad = (1/m) * (X.T @ (a-y))
    
    return grad

## Full-Hessian 

def hessian(X, y, Theta):
    m = y.size

    z = X @ Theta
    a = sigmoid(z)
    
    temp = np.diag((a * (1-a))[:,0])
    
    Hess = (1/m) * (X.T @ temp @ X)    

    return Hess


# In[164]:


# Training the model using Newton's method

def train(X, y, Theta, tolerance = 1e-10):
    # Tolerance - test for convergence
    
    Jhist = []
    i = 1
    Lambda = 50   # Regularizer factor
    
    while True:
        
        grad = gradient(X, y, Theta)
        Hess = hessian(X, y, Theta)
        
        Theta -= (np.linalg.inv(Hess + Lambda * np.eye(Hess.shape[0])) @ grad)
        
        J = cost(X, y, Theta)
        
        if i != 1 and abs(Jhist[-1] - J) < tolerance:
            print("Iteration:: {} - Cost:: {}".format(i, Jhist[-1]))
            print("Iteration:: {} - Cost:: {}".format(i, J))
            print('converged!!!')
            break
            
        Jhist.append(J)
        
        if i%1000 == 0:
            print("Iteration:: {} - Cost:: {}".format(i, J))
        
        i += 1
        
    return Theta, Jhist


# In[165]:


# Splitting dataset into test set and train set

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.3, random_state = 3)


# In[166]:


## Training our model
Theta = rand_init_weights()

Theta, Jhist = train(Xtrain,ytrain,Theta)


# In[168]:


plt.plot(Jhist, color = 'r', linewidth = 2, label = 'Cost of the model')

plt.title('Cost of the predictor vs. Number of iterations')

plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()

plt.show()


# In[172]:


# Plotting decision boundary and training data

Xplot = np.array([[np.min(X[:,1])-2], [np.max(X[:,1])+2]])
yplot = (-1/Theta[2])*(Theta[0] + Theta[1] * Xplot)

plt.plot(Xplot, yplot, label = 'Decision boundary')

for i in range(ytrain.size):
    if ytrain[i] == 0:
        plt .scatter(Xtrain[i, 1], Xtrain[i, 2], marker = '_', color = 'r', s = 45)
    else:
        plt.scatter(Xtrain[i, 1], Xtrain[i, 2], marker = '+', color = 'b', s = 45)

plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.legend()

plt.show()


# In[178]:


print("""
Expected values of Theta

b = [-25.16127]
W = [[0.20623]
 [0.20147]]

""")
print("""
Learned values of Theta
b = {}
W = {}
""".format(Theta[0], Theta[1:]))


# In[188]:


## Predicting for test dataset

predictions = np.round(sigmoid(Xtest @ Theta))

## Comparing the predictions with actual results

print('Prediction \t Ytest')
for i in range(predictions.size):
    print('   {} \t\t  {}'.format(predictions[i,0], ytest[i,0]))


# In[193]:


## Calculating accuracy of our model

p, r, f, s = precision_recall_fscore_support(ytest, predictions)

print("""Precision::\t{}
Recall::\t{}
F-Score::\t{}
Support::\t{}""".format(p, r, f, s))

print("Accuracy score::\t{}".format(accuracy_score(ytest, predictions)))


# ``~Nikhil_Chigali``

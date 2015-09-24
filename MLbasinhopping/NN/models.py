import os
import sys
import time
import gzip
import cPickle

import numpy
import numpy as np

import theano
import theano.tensor as T

from pele.potentials import BasePotential
from pele.systems import BaseSystem

from MLbasinhopping.base import BaseModel, MLSystem
from MLbasinhopping.NN.mlp import MLP

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    
    rval = [(train_set_x, train_set_y),
            (test_set_x, test_set_y), 
            (valid_set_x, valid_set_y)]
    
    return rval

def get_data():
    
    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    dataset = "mnist.pkl.gz"
    with gzip.open(os.path.join(__location__, dataset)) as f:
        train_set, valid_set, test_set = cPickle.load(f)
    
    # training data
    x = train_set[0]
    ndata, n_features = x.shape
#     
#     print "Data set size:\n", ndata, " data points\n", n_features, " features\n";
#     exit()
    # labels
    t = train_set[1]
    assert ndata == t.size
    
    # test data
    test_x = test_set[0]
    test_t = test_set[1].astype('int32')

    return x, t, test_x, test_t
        
class NNModel(BaseModel):
    def __init__(self, ndata=1000, n_hidden=10, L1_reg=0.00, L2_reg=0.0001, bias_reg=0.00):
        
        train, test, valid = load_data("/home/ab2111/source/MLbasinhopping/MLbasinhopping/NN/mnist.pkl.gz")
#         train_x = train[0][:ndata,:]
#         train_t = train[1][:ndata]
        train_x = train[0]
        train_t = train[1]
#         train_t = np.asarray(train_t, dtype="int32")

        self.train_t = train_t
        test_x = test[0]
        test_t = test[1]
        valid_x = valid[0]
        valid_t = valid[1]
        
#         test_t = np.asarray(test_t, dtype="int32")
#         valid_t = np.asarray(test_t, dtype="int32")
   
        self.train_x = train_x
        self.train_t = train_t
        self.test_x = test_x
        self.test_t = test_t
        self.valid_x = valid_x
        self.valid_t = valid_t
        
           
        print self.train_x.shape, self.train_t.shape
        print self.test_x.shape, self.test_t.shape
        print self.valid_x.shape, self.valid_t.shape
        
        print self.train_x
        
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.bias_reg = bias_reg
        
        # allocate symbolic variables for the data.  
        # Make it shared so it cab be passed only once 
        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.t = T.ivector('t')  # the labels are presented as 1D vector of

        print self.t.shape

#         x = theano.shared(value=train_x, name='x')  # the data is presented as rasterized images
#         t = theano.shared(value=train_t, name='t')  # the labels are presented as 1D vector of
                            # [int] labels
            
        rng = numpy.random.RandomState(1234)
        
        # construct the MLP class
        classifier = MLP(
            rng=rng,
            input=self.x,
            n_in=28 * 28,
            n_hidden=n_hidden,
            n_out=10
        )
        self.classifier = classifier
    
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        cost = (
            classifier.negative_log_likelihood(self.t)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr
            + bias_reg* classifier.bias_sqr
        )
        self.theano_cost = cost
        
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in classifier.params]
        self.gparams = gparams
        
        outputs = [cost] + gparams
        self.theano_cost_gradient = theano.function(
               inputs=(),
               outputs=outputs
               )
        
        # compute the errors applied to test set
        self.theano_testset_errors = theano.function(
               inputs=(),
#                outputs=self.classifier.errors(t),
                outputs=self.classifier.errors_vector(self.t),
               givens={
                       self.x: test_x,
                       self.t: test_t
                       }                                          
               )
        
        # compute the softmax output from test set
        self.theano_softmax_errors = theano.function(
               inputs=(),
#                outputs=self.classifier.errors(t),
                outputs=self.classifier.logRegressionLayer.p_y_given_x,
               givens={
                       self.x: test_x
                       }                                          
               )        
    #    res = get_gradient(train_x, train_t)
    #    print "result"
    #    print res
    #    print ""
    
        self.nparams = sum([p.get_value().size for p in classifier.params])
        self.param_sizes = [p.get_value().size for p in classifier.params]
        self.param_shapes = [p.get_value().shape for p in classifier.params]             
    
    def _cost_gradient(self):
        ret = self.theano_cost_gradient()
        cost = float(ret[0])
        gradients = ret[1:]
                
        grad = np.zeros(self.nparams)
        
        i = 0
        for g in gradients:
            npar = g.size
            grad[i:i+npar] = g.ravel()
            i += npar
        return cost, grad
        
    def get_params(self):
        params = np.zeros(self.nparams)
        i = 0
        for p in self.classifier.params:
            p = p.get_value()
            npar = p.size
            params[i:i+npar] = p.ravel()
            i += npar
        return params
    
    def set_params(self, params_vec):
        assert params_vec.size == self.nparams
        i = 0
        for count, p in enumerate(self.classifier.params):
            npar = self.param_sizes[count]
            p.set_value(params_vec[i:i+npar].reshape(self.param_shapes[count]))
            i += npar
    
    def cost(self, params):
        return self.costGradient(params)[0]
        
    def getValidationError(self, params):
        # returns the fraction of misassignments for test set
        self.set_params(params)
        return self.theano_testset_errors()
        
    def costGradient(self, params):
        # the params are stored as shared variables so we have to update
        # them in memory before computing the cost.
        self.set_params(params)
        return self._cost_gradient()

class NNSGDModel(NNModel):
    def __init__(self, batch_size = 20, learning_rate = 0.01, *args, **kwargs):
        super(NNSGDModel, self).__init__(*args, **kwargs)
        
        test_model = theano.function(
            inputs=[self.index],
            outputs=self.classifier.errors(self.t),
            givens={
                self.x: self.test_x[self.index :(self.index + 1) ],
#                 x: self.test_x[self.index * batch_size:(self.index + 1) * batch_size],
                self.t: self.test_t[self.index * batch_size:(self.index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            inputs=[self.index],
            outputs=self.classifier.errors(self.t),
            givens={
                self.x: self.valid_x[self.index * batch_size:(self.index + 1) * batch_size],
                self.t: self.valid_t[self.index * batch_size:(self.index + 1) * batch_size]
            }
        )
    
    
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
    
        # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.classifier.params, self.gparams)
        ]
    
        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[self.index],
            outputs=self.theano_cost,
            updates=updates,
            givens={
                self.x: self.train_x[self.index * batch_size: (self.index + 1) * batch_size],
                self.t: self.train_y[self.index * batch_size: (self.index + 1) * batch_size]
            }
        )
        
    def sgd(self, coords):
        class Dummy(object):
            def __init__(self, coords):
                self.coords = coords
        
        return Dummy(coords)
    
def myquench(coords, pot, tol, **kwargs):
    """ This quench checks the rmsgrad condition is satisfied, and keeps quenching until this is the case
    """
    from pele.optimize import lbfgs_cpp as quench

    rmsgrad = 10000.0 

    while rmsgrad > tol:
        ret = quench(coords, pot, tol=tol, **kwargs)
        e,g = pot.getEnergyGradient(ret.coords)
        rmsgrad = np.linalg.norm(g)  
        coords = ret.coords
        
    return ret

def quench_with_sgd(coords, model, lbfgs_quench):
    
    sgd_ret = model.sgd(coords)
    
    return lbfgs_quench(sgd_ret.coords)
    
    
class NNSystem(MLSystem):
    def __init__(self, model, minimizer_tolerance=1.0e-5, *args, **kwargs):
        super(NNSystem, self).__init__(model)
        self.minimizer_tolerance = minimizer_tolerance

    def get_minimizer(self, nsteps=1e5, M=4, iprint=0, maxstep=1.0, **kwargs):
        from pele.optimize import lbfgs_cpp
        
        lbfgs_quench = lambda coords: lbfgs_cpp(coords, self.get_potential(), tol=self.minimizer_tolerance, 
                                     nsteps=nsteps, M=M, iprint=iprint, 
                                     maxstep=maxstep, 
                                     **kwargs)
        if hasattr(self.model, "sgd"):
            return lambda coords: quench_with_sgd(coords, self.model, lbfgs_quench)
        
        else:
            return lbfgs_quench
    
def test():
    
    model = NNModel()
    system = NNSystem(model)
    t = system.get_potential()
    
    nparams = model.nparams        
    print "get_energy_gradient"
    newparams = np.random.uniform(-.05, .05, nparams)
    e, g = t.getEnergyGradient(newparams)
    print "cost", e
    
    print "\n\nagain\nget_energy_gradient"
    newparams = np.random.uniform(-.05, .05, nparams)
    e, g = t.getEnergyGradient(newparams)
    print "cost", e
    
    params = model.get_params()
    print params
    dx = np.max(np.abs(params - newparams))
    print dx
    assert dx < 1e-8
    
    # do minimization
    from pele.optimize import lbfgs_py
    res = lbfgs_py(newparams, t, iprint=10, tol=1e-4)
    print res

if __name__ == "__main__":
    test()
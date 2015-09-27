import numpy as np
import unittest
import logging
import sys

from MLbasinhopping.utils import run_basinhopping, run_double_ended_connect, make_disconnectivity_graph

from MLbasinhopping.NN.models import NNSystem, NNModel, NNSGDModel

class CheckCorrectOutput(unittest.TestCase):

    def setUp(self):
        
        np.random.seed(12345)
        
        ndata = 10000
#         n_hidden = 10
        n_hidden = 500
        p = 2
        L2_reg=np.power(1.0e1, -p)
    #     L1_reg=np.power(1.0e1, -p)
    #     L2_reg=0.0
        L1_reg=0.0
        bias_reg = 0.0
        
#         model = NNSGDModel(ndata=ndata, n_hidden=n_hidden, L1_reg=L1_reg, L2_reg=L2_reg, bias_reg=bias_reg)
        model = NNModel(ndata=ndata, n_hidden=n_hidden, L1_reg=L1_reg, L2_reg=L2_reg, bias_reg=bias_reg)
        self.system = NNSystem(model)

        self.db = self.system.create_database()
    
    def test_minimizer(self):
        
        quench = self.system.get_minimizer(iprint=100)
#         coords = np.random.random(self.system.model.nparams)
        Nparams = self.system.model.nparams

        coords = np.random.uniform(
                    low=-np.sqrt(6. / Nparams),
                    high=np.sqrt(6. / Nparams),
                    size=Nparams
                    )
        ret = quench(coords)
        self.assertAlmostEqual(ret.energy, 3.02478820124, msg="Error: Quenched energy = "+str(ret.energy))
           
    def test_initial_energy(self):
       
        pot = self.system.get_potential()

        Eref=1493.6103675
        E = pot.getEnergy(np.random.random(self.system.model.nparams))
        self.assertAlmostEqual(E, Eref,
                               msg="Energy " + str(E) + " != "+str(Eref))    
#         run_basinhopping(system, nsteps, db)     

        # connect minima
#         run_double_ended_connect(system, db)
            
        # connect minima
#         make_disconnectivity_graph(system, db)



#     def test_costGradient(self):
#         
#         pot = self.system.get_potential()
#         coords = np.random.random(self.system.model.params.get_value().shape)
#         
#         c, g = pot.getEnergyGradient(coords)
#         
#         self.assertAlmostEqual(c, 126.055885792)
#         self.assertAlmostEqual(g[0], 259.29652069)
#         
#         log = logging.getLogger("CheckCorrectOutput.test_costGradient")
#         log.debug("cost, grad=\n"+str(c)+"\n"+str(g))
        
logging.basicConfig(stream=sys.stderr)
logging.getLogger("CheckCorrectOutput.test_costGradient").setLevel(logging.DEBUG)
suite1 = unittest.TestLoader().loadTestsFromTestCase(CheckCorrectOutput)
# suite2 = unittest.TestLoader().loadTestsFromTestCase(FindingInterestingParamsTestCase)

suite = unittest.TestSuite(tests=[suite1])
unittest.TextTestRunner(verbosity=2).run(suite)
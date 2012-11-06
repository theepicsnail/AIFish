from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet

ds = SupervisedDataSet(2,1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

net = buildNetwork(2, 3, 1)

trainer = BackpropTrainer(net, ds)

ROW = "{:8}{:8} | {:<8} | {:<8}"

def test(net,ds):
    print ROW.format("In1","In2","Expected","Actual")
    for params,result in ds:
        val = net.activate(params)[0] > .5
        p1 = params[0] > .5
        p2 = params[1] > .5
        r = result[0] > .5
        print ROW.format(p1,p2,r,val)

print "Before training:"
test(net,ds)

print "Training"
for i in xrange(10000):
    trainer.train()

print "After training:"
test(net,ds)

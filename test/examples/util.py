import numpy as np
from chainer import Chain, report, Variable, links as L
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam
from chainer.training import Trainer, StandardUpdater, extensions
from chainerltr import Ranker
from chainerltr.dataset import zeropad_concat
from test.test_dataset import get_dataset


class Loss(Chain):
    def __init__(self, ranker, loss_fn):
        super().__init__(ranker=ranker)
        self.loss_fn = loss_fn

    def __call__(self, xs, ys, nr_docs):
        loss = self.loss_fn(self.ranker(xs), ys, nr_docs)
        report({"loss": loss})
        return loss


def run_linear_network(loss_fn, alpha=0.3, batch_size=2):

    # Get data
    np.random.seed(42)
    dataset = get_dataset()
    iterator = SerialIterator(dataset, batch_size, repeat=True, shuffle=True)

    # Set up network and loss
    predictor = L.Linear(None, 1)
    ranker = Ranker(predictor)
    loss = Loss(ranker, loss_fn)

    # Optimizer
    optimizer = Adam(alpha=alpha)
    optimizer.setup(loss)
    updater = StandardUpdater(iterator, optimizer, converter=zeropad_concat)
    trainer = Trainer(updater, (100, 'epoch'))
    log_report = extensions.LogReport(log_name=None)
    trainer.extend(log_report)
    np.random.seed(42)
    trainer.run()
    last_loss = log_report.log[-1]['loss']
    return last_loss

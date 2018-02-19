from chainerltr.loss.listwise import listnet, listmle, listpl
from chainerltr.loss.pairwise import ranknet
from test.examples.util import run_linear_network
from chainer.testing import assert_allclose, condition


def test_linear_listnet(alpha=0.3, batch_size=2):
    final_loss = run_linear_network(listnet, alpha, batch_size)
    print(final_loss)
    assert_allclose(final_loss, 0.20512548089027405, atol=1e-2, rtol=1e-2)


def test_linear_listmle(alpha=0.85, batch_size=1):
    final_loss = run_linear_network(listmle, alpha, batch_size)
    print(final_loss)
    assert_allclose(final_loss, 4.400527318318685, atol=1e-2, rtol=1e-2)


def test_linear_listpl(alpha=0.15, batch_size=1):
    final_loss = run_linear_network(listpl, alpha, batch_size)
    assert_allclose(final_loss, 9.079656283060709, atol=1e-2, rtol=1e-2)


def test_linear_ranknet(alpha=0.5, batch_size=1):
    final_loss = run_linear_network(ranknet, alpha, batch_size)
    assert_allclose(final_loss, 0.5795594851175944, atol=1e-2, rtol=1e-2)

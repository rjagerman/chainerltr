from chainerltr.loss.listwise import listnet, listmle, listpl
from chainerltr.loss.pairwise import ranknet
from test.examples.util import run_linear_network
from chainer.testing import assert_allclose, condition


def test_linear_listnet(alpha=0.05, batch_size=1):
    final_ndcg = run_linear_network(listnet, alpha, batch_size)
    assert_allclose(final_ndcg, 0.9889815968059534, atol=1e-2, rtol=1e-2)


def test_linear_listmle(alpha=1.73, batch_size=1):
    final_ndcg = run_linear_network(listmle, alpha, batch_size)
    assert_allclose(final_ndcg, 0.9910497691677081, atol=1e-2, rtol=1e-2)


def test_linear_listpl(alpha=0.30, batch_size=2):
    final_ndcg = run_linear_network(listpl, alpha, batch_size)
    assert_allclose(final_ndcg, 0.9833011615771335, atol=1e-2, rtol=1e-2)


def test_linear_ranknet(alpha=0.5, batch_size=1):
    final_ndcg = run_linear_network(ranknet, alpha, batch_size)
    assert_allclose(final_ndcg, 0.9956287397900928, atol=1e-2, rtol=1e-2)

import warnings
import numpy as np
import torch

from asdfghjkl import FISHER_EXACT, FISHER_MC, COV
from asdfghjkl import SHAPE_KRON, SHAPE_DIAG
from asdfghjkl import fisher_for_cross_entropy
from asdfghjkl.gradient import batch_gradient

from laplace.curvature import CurvatureInterface
from laplace.matrix import Kron
from laplace.utils import is_batchnorm


class KazukiInterface(CurvatureInterface):

    def __init__(self, model, likelihood, last_layer=False):
        if likelihood != 'classification':
            raise ValueError('This backend does only support classification currently.')
        self.last_layer = last_layer
        super().__init__(model, likelihood)

    @staticmethod
    def jacobians(model, X):
        Js = list()
        for i in range(model.output_size):
            def loss_fn(outputs, targets):
                return outputs[:, i].sum()

            f = batch_gradient(model, loss_fn, X, None).detach()
            Js.append(_get_batch_grad(model))
        Js = torch.stack(Js, dim=1)
        return Js, f

    @property
    def ggn_type(self):
        raise NotImplementedError()

    def _get_kron_factors(self, curv, M):
        kfacs = list()
        for module in curv._model.modules():
            if is_batchnorm(module):
                warnings.warn('BatchNorm unsupported for Kron, ignore.')
                continue

            stats = getattr(module, self.ggn_type, None)
            if stats is None:
                continue
            if hasattr(module, 'bias') and module.bias is not None:
                # split up bias and weights
                kfacs.append([stats.kron.B, stats.kron.A[:-1, :-1]])
                kfacs.append([stats.kron.B * stats.kron.A[-1, -1] / M])
            elif hasattr(module, 'weight'):
                p, q = np.prod(stats.kron.B.shape), np.prod(stats.kron.A.shape)
                if p == q == 1:
                    kfacs.append([stats.kron.B * stats.kron.A])
                else:
                    kfacs.append([stats.kron.B, stats.kron.A])
            else:
                raise ValueError(f'Whats happening with {module}?')
        return kfacs

    @staticmethod
    def _rescale_kron_factors(kron, N):
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= 1/N
        return kron

    def diag(self, X, y, **kwargs):
        curv = fisher_for_cross_entropy(self.model, self.ggn_type, SHAPE_DIAG, inputs=X, targets=y)
        diag_ggn = curv.matrices_to_vector(None)
        with torch.no_grad():
            loss = self.lossfunc(self.model(X), y)
        return self.factor * loss, self.factor * diag_ggn

    def kron(self, X, y, N, **wkwargs) -> [torch.Tensor, Kron]:
        M = len(y)
        curv = fisher_for_cross_entropy(self.model, self.ggn_type, SHAPE_KRON, inputs=X, targets=y)
        kron = Kron(self._get_kron_factors(curv, M))
        kron = self._rescale_kron_factors(kron, N)
        with torch.no_grad():
            loss = self.lossfunc(self.model(X), y)
        return self.factor * loss, self.factor * kron

    def full(self, X, y, **kwargs):
        raise NotImplementedError()


class KazukiGGN(KazukiInterface):

    def __init__(self, model, likelihood, last_layer=False, stochastic=False):
        super().__init__(model, likelihood, last_layer)
        self.stochastic = stochastic

    @property
    def ggn_type(self):
        return FISHER_MC if self.stochastic else FISHER_EXACT


class KazukiEF(KazukiInterface):

    @property
    def ggn_type(self):
        return COV


def _get_batch_grad(model):
    batch_grads = list()
    for module in model.modules():
        if hasattr(module, 'op_results'):
            res = module.op_results['batch_grads']
            if 'weight' in res:
                batch_grads.append(res['weight'].flatten(start_dim=1))
            if 'bias' in res:
                batch_grads.append(res['bias'].flatten(start_dim=1))
            if len(set(res.keys()) - {'weight', 'bias'}) > 0:
                raise ValueError(f'Invalid parameter keys {res.keys()}')
    return torch.cat(batch_grads, dim=1)

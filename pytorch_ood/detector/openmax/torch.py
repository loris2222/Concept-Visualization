import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ...api import Detector
from ...utils import TensorBuffer, is_known

log = logging.getLogger(__name__)


class OpenMax(Detector):
    """
    Implementation of the OpenMax Layer as proposed in the paper *Towards Open Set Deep Networks*.

    The methods determines a center :math:`\\mu_y` for each class in the logits space of a model, and then
    creates a statistical model of the distances of correct classified inputs.
    It uses extreme value theory to detect outliers by fitting a weibull function to the tail of the distance
    distribution.

    We use the activation of the *unknown* class as outlier score.

    .. warning:: This methods requires ``libmr`` to be installed, which is broken at the moment. You can only use it
       by installing ``cython`` and ``numpy``, and ``libmr`` manually afterwards.

    :see Paper: `ArXiv <https://arxiv.org/abs/1511.06233>`__
    :see Implementation: `GitHub <https://github.com/abhijitbendale/OSDN>`__
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tailsize: int = 25,
        alpha: int = 10,
        euclid_weight: float = 1.0,
    ):
        """
        :param tailsize: length of the tail to fit the distribution to
        :param alpha: number of class activations to revise
        :param euclid_weight: weight for the euclidean distance.
        """
        self.model = model

        # we import it here because of its dependency to the broken libmr
        from .numpy import OpenMax as NPOpenMax

        self.openmax = NPOpenMax(tailsize=tailsize, alpha=alpha, euclid_weight=euclid_weight)

    def fit(self, data_loader: DataLoader, device: Optional[str] = "cpu"):
        """
        Determines parameters of the weibull functions for each class.

        :param data_loader: Data to use for fitting
        :param device: Device used for calculations
        """
        z, y = OpenMax._extract(data_loader, self.model, device=device)
        self.openmax.fit(z.numpy(), y.numpy())

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: class logits
        """
        with torch.no_grad():
            z = self.model(x).cpu().numpy()

        return torch.tensor(self.openmax.predict(z)[:, 0])

    def get_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: class logits
        """
        with torch.no_grad():
            z = self.model(x).cpu().numpy()

        return torch.tensor(self.openmax.predict(z))

    @staticmethod
    def _extract(data_loader, model: torch.nn.Module, device):
        """
        Extract embeddings from model

        .. note :: this should be moved into a dedicated function

        :param data_loader:
        :param model:
        :return:
        """
        buffer = TensorBuffer()
        log.debug("Extracting features")
        for batch in data_loader:
            x = batch['X'].float()
            y = batch['y'].long().squeeze()
            # x, y = batch
            # x = x.to(device)
            known = is_known(y)
            z = model(x[known])
            # flatten
            x = z.view(x.shape[0], -1)
            buffer.append("embedding", z[known])
            buffer.append("label", y[known])

        z = buffer.get("embedding")
        y = buffer.get("label")

        buffer.clear()
        return z, y

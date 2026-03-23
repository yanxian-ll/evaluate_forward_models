from __future__ import annotations

import hydra

from mapanything.models import MODEL_CONFIGS
from mapanything.train import training as training_module
from mapanything.train.training import train

from mapanything.models.external.vggt_midmatch import VGGTMidMatchWrapper
from mapanything.loss.custom_midmatch_loss import VGGTMidMatchCriterion


def register_midmatch_extensions() -> None:
    MODEL_CONFIGS["vggt_midmatch"] = {"class": VGGTMidMatchWrapper}
    training_module.__dict__["VGGTMidMatchCriterion"] = VGGTMidMatchCriterion


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    register_midmatch_extensions()
    train(cfg)


if __name__ == "__main__":
    main()

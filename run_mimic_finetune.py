from templates_cls import *
from experiment_classifier_funetune import *

if __name__ == '__main__':
    # need to first train the diffae autoencoding model & infer the latents
    # this requires only a single GPU.
    gpus = [6]
    conf = mimic256_autoenc_cls_finetune()
    train_cls(conf, gpus=gpus)

    # after this you can do the manipulation!

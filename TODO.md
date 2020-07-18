- weight_decay: 0.0005, 0.001, 0.01
- AGW: [Paper](https://arxiv.org/pdf/2001.04193.pdf), [Code](https://github.com/mangye16/ReID-Survey)
- ABD-Net: [Paper](https://arxiv.org/pdf/1908.01114.pdf), [Code](https://github.com/TAMU-VITA/ABD-Net)
- AdaptiveReID: [Paper](https://arxiv.org/pdf/2007.07875v1.pdf), [Code](https://github.com/nixingyang/AdaptiveReID)
- SBS: [Paper](https://arxiv.org/pdf/2006.02631.pdf), [Code](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md)
- MGN: [Paper](https://arxiv.org/pdf/1804.01438v1.pdf), [Code](https://github.com/GNAYUOHZ/ReID-MGN)
- support more backbone: [resnet50_ibn_a](https://github.com/XingangPan/IBN-Net), ...

- build_backbone function.
- build_transforms function.

- transforms: [Auto-augment](https://github.com/JDAI-CV/fast-reid/blob/ee634df2900996233473cb95a80029bd456cce97/fastreid/data/transforms/autoaugment.py#L495), [Random patch](https://github.com/JDAI-CV/fast-reid/blob/ee634df290/fastreid/data/transforms/transforms.py),

- Pooling: [GeM pooling](https://github.com/JDAI-CV/fast-reid/blob/46228ce946/fastreid/layers/gem_pool.py), [Attention Pooling](https://github.com/JDAI-CV/fast-reid/blob/46228ce946/fastreid/layers/attention.py)

- [Reduction head](https://github.com/JDAI-CV/fast-reid/blob/46228ce946/fastreid/modeling/heads/reduction_head.py)

- ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)

- Non-Local Resnet

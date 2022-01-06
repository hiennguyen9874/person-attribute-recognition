import torch.optim as optim


def build_optimizers(config, model):
    assert "optimizer" in config
    cfg_optimizer = config["optimizer"]

    assert "name" in cfg_optimizer
    assert "lr" in cfg_optimizer

    specified_lr = False
    if (
        "specified_lr" in cfg_optimizer
        and cfg_optimizer["specified_lr"]["enable"] == True
    ):
        assert "layers" in cfg_optimizer["specified_lr"]
        assert "lr" in cfg_optimizer["specified_lr"]

        specified_lr = True
        base_params = []
        new_params = []
        for name, module in model.named_children():
            if name not in cfg_optimizer["specified_lr"]["layers"]:
                base_params += [p for p in module.parameters()]
            else:
                new_params += [p for p in module.parameters()]
        param_groups = [
            {"params": base_params},
            {"params": new_params, "lr": cfg_optimizer["specified_lr"]["lr"]},
        ]
    else:
        param_groups = model.parameters()

    base_learning_rate = cfg_optimizer["lr"]
    dict_paramsters = {"lr": base_learning_rate}
    if specified_lr:
        dict_paramsters.update(
            {
                "specified_lr (%f)"
                % cfg_optimizer["specified_lr"]["lr"]: cfg_optimizer["specified_lr"][
                    "layers"
                ]
            }
        )

    if cfg_optimizer["name"] == "adam":
        dict_paramsters.update(
            {
                "weight_decay": cfg_optimizer["weight_decay"],
                "beta1": cfg_optimizer["beta1"],
                "beta2": cfg_optimizer["beta2"],
            }
        )

        return (
            optim.Adam(
                param_groups,
                lr=base_learning_rate,
                weight_decay=cfg_optimizer["weight_decay"],
                betas=(cfg_optimizer["beta1"], cfg_optimizer["beta2"]),
            ),
            dict_paramsters,
        )

    elif cfg_optimizer["name"] == "sgd":
        dict_paramsters.update(
            {
                "momentum": cfg_optimizer["momentum"],
                "weight_decay": cfg_optimizer["weight_decay"],
                "dampening": cfg_optimizer["dampening"],
                "nesterov": cfg_optimizer["nesterov"],
            }
        )

        return (
            optim.SGD(
                param_groups,
                lr=base_learning_rate,
                momentum=cfg_optimizer["momentum"],
                weight_decay=cfg_optimizer["weight_decay"],
                dampening=cfg_optimizer["dampening"],
                nesterov=cfg_optimizer["nesterov"],
            ),
            dict_paramsters,
        )
    else:
        raise KeyError("config[optimizer][name] error")

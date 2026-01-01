import typing as tp
import omegaconf

def dict_from_config(cfg: tp.Any) -> tp.Any:
    """
    Convert an omegaconf config to a python dict, resolving variable interpolation.
    """
    if isinstance(cfg, omegaconf.DictConfig):
        return omegaconf.OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, omegaconf.ListConfig):
        return omegaconf.OmegaConf.to_container(cfg, resolve=True)
    return cfg

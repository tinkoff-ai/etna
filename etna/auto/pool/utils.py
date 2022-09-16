from etna.commands import OmegaConf


def fill_template(config: dict, interpolate_config: dict) -> dict:
    """Fill given ``config`` with values from ``interpolate_config``."""
    temp_config = OmegaConf.create(config)
    _interpolate_config = OmegaConf.create({"__aux__": interpolate_config})
    temp_config.merge_with(_interpolate_config)
    filled_config: dict = OmegaConf.to_container(temp_config, resolve=True)  # type: ignore
    del filled_config["__aux__"]
    return filled_config

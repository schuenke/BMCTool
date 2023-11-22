"""set_params.py Functions to load the parameters for the simulation from the
config files."""
import copy
from pathlib import Path

import yaml

from src.bmctool.params import Params


def load_config(*args: str | Path) -> dict:
    """load_config Load the config file(s) for given path(s) and return the
    data as a dictionary.

    Returns
    -------
    dict
        Dictionary containing the data from the config file(s).
    """
    config = {}
    for filepath in args:
        with open(filepath) as file:
            config.update(yaml.load(file, Loader=yaml.Loader))
    return config


def check_values(
    val_dict: dict, config: dict, invalid: list, dict_key: str = None, reference_config: str | Path = None
) -> tuple[dict, list]:
    """check_values Check and correct the nested dictionaries from the loaded
    configuration for definition errors.

    Returns
    -------
    Tuple[dict, list]
        Tuple containing the corrected config dictionary and a list of invalid parameters.
    """
    if reference_config is None:
        reference_config = Path(__file__).parent / 'library' / 'maintenance' / 'valid_params.yaml'
    valids = load_config(reference_config)
    valid_num = valids['valid_num']
    valid_str = valids['valid_str']
    valid_bool = valids['valid_bool']
    valid_lineshapes = valids['valid_lineshapes']
    val_dict = copy.deepcopy(val_dict)  # create a deep copy of val_dict before iterating over it
    for k, v in val_dict.items():
        if k not in valid_num + valid_str + valid_bool:
            if dict_key:
                invalid.append(k + ' in ' + dict_key)
            else:
                invalid.append(k)
        elif k in valid_str:
            if type(v) is not str:
                if dict_key:
                    invalid.append(str(v) + ' value from ' + k + ' in ' + dict_key + ' should be of type string.')
                else:
                    invalid.append(str(v) + ' value from ' + k + ' should be of type string.')
        elif k in valid_num:
            if type(v) not in [float, int]:
                if type(v) is str:
                    if type(eval(v)) in [float, int]:
                        if dict_key:
                            config[dict_key][k] = eval(v)
                        else:
                            config[k] = eval(v)
                    else:
                        if dict_key:
                            invalid.append(
                                str(v) + ' value from ' + k + ' in ' + dict_key + ' should be a numerical type.'
                            )
                        else:
                            invalid.append(str(v) + ' value from ' + k + ' should be a numerical type.')
            if k == 't1':
                config[dict_key]['r1'] = 1 / config[dict_key].pop(k)
            elif k == 't2':
                config[dict_key]['r2'] = 1 / config[dict_key].pop(k)
        elif k in valid_bool:
            if type(v) is not bool:
                if v in ['true', 'True', 'TRUE', 'yes', 'Yes', 'yes', 1]:
                    if dict_key:
                        config[dict_key][k] = True
                    else:
                        config[k] = True
                elif v in ['false', 'False', 'FALSE', 'no', 'No', 'NO', 0]:
                    if dict_key:
                        config[dict_key][k] = False
                    else:
                        config[k] = False
                else:
                    if dict_key:
                        invalid.append(str(v) + ' value from ' + k + ' in ' + dict_key + ' should be of type bool.')
                    else:
                        invalid.append(str(v) + ' value from ' + k + ' should be of type bool.')
        if k == 'seq_file':
            if not Path(v).exists():
                invalid.append('Seq_file leads to an invalid path.')
        if k == 'lineshape':
            if v not in valid_lineshapes:
                if v in ['None', 'none', 'NONE', False, 'False', 'false', 'FALSE', 'no', 'No', 'NO']:
                    config[dict_key][k] = None
                else:
                    invalid.append(
                        str(v)
                        + ' value from '
                        + k
                        + ' in '
                        + dict_key
                        + ' should be out of: '
                        + ''.join(x + ', ' for x in valid_lineshapes[:-1])
                        + valid_lineshapes[-1]
                    )
    return config, invalid


def check_cest_values(val_dict: dict, config: dict, invalid: list, dict_key: str) -> tuple[dict, list]:
    """check_cest_values Check and correct the cest pool parameters for
    definition errors.

    Returns
    -------
    Tuple[dict, list]
        Tuple containing the corrected config dictionary and a list of invalid parameters.
    """
    if 'cest_pools' in config.keys():
        config['cest_pool'] = config.pop('cest_pools')
    conf_temp = config['cest_pool']
    conf_temp, invalid_new = check_values(val_dict, conf_temp, invalid, dict_key)
    if invalid_new != invalid:
        invalid_new.append('some definition in cest_pool')
    config['cest_pool'] = conf_temp
    return config, invalid


def check_necessary(config: dict, necessary: list, necessary_w: list = None):
    """check_necessary Check for necessary parameters in the loaded values.

    Parameters
    ----------
    config : dict
        Dictionary containing the loaded configuration values
    necessary : list
        List of required parameters
    necessary_w : list, optional
        List of required parameters of the water pool, by default None

    Raises
    ------
    AssertionError
        If a necessary parameter is missing
    """
    missing = []
    for n in necessary:
        if n not in config.keys():
            missing.append(n)
    if missing:
        raise AssertionError(
            'The following parameters have to be defined: ' + ''.join(m + ', ' for m in missing[:-1]) + missing[-1]
        )
    if necessary_w:
        for i in range(2):
            if (
                necessary_w[0][i] not in config['water_pool'].keys()
                and necessary_w[1][i] not in config['water_pool'].keys()
            ):
                missing.append('t' + str(i + 1) + ' or r' + str(i + 1))
    if missing:
        raise AssertionError(
            'The following water_pool parameters have to be defined: '
            + ''.join(m + ', ' for m in missing[:-1])
            + missing[-1]
        )


def check_params(config: dict, reference_config: str | Path = None) -> dict:
    """check_params Check and correct the loaded parameters.

    Parameters
    ----------
    config : dict
        Dictionary containing the loaded configuration values
    reference_config : Union[str, Path], optional
        Path to the reference config, by default None

    Returns
    -------
    dict
        Dictionary containing the corrected configuration values

    Raises
    ------
    AssertionError
        If an invalid parameter is found or a necessary parameter is missing
    """
    if reference_config is None:
        reference_config = Path(__file__).parent / 'library' / 'maintenance' / 'valid_params.yaml'
    invalid = []
    valids = load_config(reference_config)
    valid = valids['valid_first']
    valid_dict = valids['valid_dict']
    valid_list = valids['valid_list']
    necessary = valids['necessary']
    if 'necessary_w' in valids.keys():
        necessary_w = valids['necessary_w']
    else:
        necessary_w = None
    if 'b0_inhomogeneity' in config.keys():
        config['b0_inhom'] = config.pop('b0_inhomogeneity')
    check_necessary(config=config, necessary=necessary, necessary_w=necessary_w)
    config_dicts = {ck: cv for ck, cv in config.items() if type(cv) is dict}
    config_lists = {ck: cv for ck, cv in config.items() if type(cv) is list}
    config_vals = {ck: cv for ck, cv in config.items() if type(cv) is not dict and type(cv) is not list}
    for k, v in config.items():
        if k not in valid:
            invalid.append(k)
    for k, v in config_dicts.items():
        if type(v[list(v.keys())[0]]) is dict:
            if k in valid_dict:
                for dk, dv in v.items():
                    config, invalid = check_cest_values(dv, config, invalid, dk)
            else:
                invalid.append(k)
        else:
            config, invalid = check_values(v, config, invalid, dict_key=k)
    for k, v in config_lists.items():
        if k not in valid_list:
            invalid.append(k)
    config, invalid = check_values(config_vals, config, invalid, reference_config=reference_config)
    if invalid:
        raise AssertionError(
            'Check parameter configuration files! \n '
            'Invalid: ' + ''.join(str(i) + ', ' for i in invalid[:-1]) + str(invalid[-1])
        )
    return config


def load_params(*filepaths: str | Path) -> Params:
    """load_params Load parameters from given path(s) into Params object.

    Returns
    -------
    Params
        Params object containing the loaded parameters

    Raises
    ------
    ValueError
        If no filepath is given
    FileExistsError
        If a given filepath does not exist
    """
    if not filepaths:
        raise ValueError('You need to define at least one filepath to configure the parameters.')
    paths = [Path(filepath) for filepath in filepaths]
    # raise error if one of the files does not exist
    for path in paths:
        if not path.exists():
            raise FileExistsError(f'File {path} does not exist.')

    # load the configurations from the files
    config = load_config(*paths)
    # check parameters for missing, typos, wrong assignments
    config = check_params(config)
    config['filepaths'] = [p.name for p in paths]
    # instantiate class to store the parameters
    sp = Params()

    # scanner and inhomogeneity settings
    sp.set_scanner(**{k: v for k, v in config.items() if k in ['b0', 'gamma', 'b0_inhom', 'rel_b1']})

    # optional params
    sp.set_options(
        **{
            k: v
            for k, v in config.items()
            if k in ['reset_init_mag', 'max_pulse_samples', 'scale', 'par_calc', 'verbose']
        }
    )
    # water pool settings
    sp.set_water_pool(**config['water_pool'])

    # cest pool settings
    if 'cest_pool' in config.keys() and config['cest_pool']:
        for pool_name, pool_params in config['cest_pool'].items():
            sp.set_cest_pool(**pool_params)

    # mt_pool settings
    if 'mt_pool' in config.keys() and config['mt_pool']:
        sp.set_mt_pool(**config['mt_pool'])

    # set m_vec
    sp.set_m_vec()

    # print configuration settings
    if sp.options['verbose']:
        sp.print_settings()

    return sp

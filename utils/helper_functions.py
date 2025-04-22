import configparser
import ast


def parse_ini_file(file_path):
    """
    Parses an INI file and returns a dictionary with sections as keys and
    dictionaries of key-value pairs as values.
    """
    config = configparser.ConfigParser()
    config.read(file_path)

    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config.items(section):
            # Check if the value is a list or a dictionary
            if value.startswith('[') and value.endswith(']'):
                config_dict[section][key] = ast.literal_eval(value)
            elif value.startswith('{') and value.endswith('}'):
                config_dict[section][key] = ast.literal_eval(value)
            # Check if the value is a number
            elif value.isdigit():
                config_dict[section][key] = int(value)
            elif value.replace('.', '', 1).isdigit():
                config_dict[section][key] = float(value)
            # Check if the value is a boolean
            elif value in ['True','true','yes','y']:
                config_dict[section][key] = True
            elif value in ['False','false','no','n']:
                config_dict[section][key] = False
            # Check if the value is None
            elif value in ['None','none','']:
                config_dict[section][key] = None
            else:
                # it's just a string
                config_dict[section][key] = value
    MainSettings = config_dict.get('MainSettings')
    RealObservations = config_dict.get('RealObservations')
    MockObservations = config_dict.get('MockObservations')
    Flows = config_dict.get('Flows')
    ExtraOptions = config_dict.get('ExtraOptions')
    return MainSettings, RealObservations, MockObservations, Flows, ExtraOptions
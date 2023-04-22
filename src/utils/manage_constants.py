import json
import sys


def get_constants(names):
    """
    Function to obtain values of constants relevant to program execution

    Inputs
    names: str or list of str containing names of constants to be retrieved

    Outputs
    constants: dict containing constant name and value as key-value pairs
    {
        constant name: constant value
    }
    """
    constants = {}

    with open('./src/constants.json', 'r') as f:
        all_constants = json.load(f)

    if type(names) == str:
        constants[names] = all_constants[names]
    elif type(names) == list:
        for name in names:
            constants[name] = all_constants[name]
    else:
        sys.exit('Unexpected argument for names. Please check value')

    return constants


def create_constants(constants):
    """
    Function to create values of new constants relevant to program execution

    Inputs
    constants: dict like containing name and value of constant

    Outputs
    None
    """
    with open('./src/constants.json', 'r') as f:
        all_constants = json.load(f)

    for constant in constants.keys():
        if constant in all_constants.keys():
            sys.exit('Constant with this name already exists. Choose another name or use update_constant instead')
        else:
            all_constants[constant] = constants[constant]
            

    with open('./src/constants.json', 'w') as f:
        json.dump(all_constants, f)

    return 0


def update_constants(constants):
    """
    Function to update values of existing constants relevant to program execution

    Inputs
    constants: dict like containing name and value of constant

    Outputs
    None
    """
    with open('./src/constants.json', 'r') as f:
        all_constants = json.load(f)

    for constant in constants.keys():
        if constant in all_constants.keys():
            all_constants[constant] = constants[constant]
        else:
            sys.exit('Constant does not exist. Use create_constant instead')

    with open('./src/constants.json', 'w') as f:
        json.dump(all_constants, f)

    return 0


def delete_constants(names, keep_name=False):
    """
    Function to delete values of constants relevant to program execution

    Inputs
    names: str or list of str containing names of constants to be removed
    keep_name (optional): Boolean. True if only value needs to be erased

    Outputs
    None
    """
    with open('./src/constants.json', 'r') as f:
        all_constants = json.load(f)

    if type(names) == str:
        if keep_name:
            all_constants[names] = None
        else:
            all_constants.pop(names)
    elif type(names) == list:
        if keep_name:
            for name in names:
                all_constants[name] = None
        else:
            for name in names:
                all_constants.pop(name)
    else:
        sys.exit('Unexpected argument for names. Please check value')

    with open('./src/constants.json', 'w') as f:
        json.dump(all_constants, f)

    return 0
class Config(object):
    """
    Class that loads options from a config file and converts
    them into attributes.
    """
    def __init__(self, filename, required_attributes=()):
        config = None
        print(filename)
        config_file = open(filename)
        config_text = config_file.read()
        config_file.close()
        if "config" not in config_text:
            raise ValueError
        d = {}
        exec(config_text, d)
        config = d["config"]
        print(config)
        for a, v in config.items():
            print(a)
            setattr(self, a, v)
        if len(required_attributes) > 0:
            has_required = [hasattr(self, a) for a in required_attributes]
            if not all(has_required):
                for i in range(len(required_attributes)):
                    if not has_required[i]:
                        print("{0} not found.".format(required_attributes[i]))
                        exit(1)


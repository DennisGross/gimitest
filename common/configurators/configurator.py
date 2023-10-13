class Configurator:

    def __init__(self, env, parameter_str, state_str):
        self.parameter_str = parameter_str
        self.state_str = state_str
        self.parameters_dict = self.__parse_str(self.parameter_str)
        self.env_state_dict = self.__parse_state_str(env, self.state_str)
        self.THRESHOLD = 10000000000

    def __parse_str(self, s):
        # paratemer_str = "parameter1:int:0:10,parameter2:float:0.0:1.0,..."
        # Return {"parameter1": {"type": "int", "low": 0, "high": 10}, "parameter2": {"type": "float", "low": 0.0, "high": 1.0}, ...}
        parameter_dict = {}
        parameter_list = s.split(",")
        for parameter in parameter_list:
            if parameter == "":
                continue
            parameter_name, parameter_type, parameter_low, parameter_high = parameter.split(":")
            parameter_dict[parameter_name] = {"type": parameter_type, "low": parameter_low, "high": parameter_high}
        return parameter_dict

    def __parse_state_str(self, env, state_str):
        user_state_dict = self.__parse_str(state_str)
        env_state_dict = env.get_state_dict()
        # is user_state_dict {}
        if len(user_state_dict.keys()) == 0:
            return env_state_dict
        else:
            env_state_dict = user_state_dict
        return env_state_dict

    def generate_state(self):
        raise NotImplementedError("Generate_state not implemented")


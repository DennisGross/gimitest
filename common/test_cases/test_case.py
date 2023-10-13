class TestCase():

    def __init__(self, test_str):
        # test_str "TEST_NAME;FORMULA"
        self.test_name = test_str.split(";")[0]
        self.formula = test_str.split(";")[1]


    def check_trajectory(self, trajectory):
        raise NotImplementedError("Test case not implemented")

    def run(self):
        raise NotImplementedError("Test case not implemented")


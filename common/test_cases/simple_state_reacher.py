from common.test_cases.test_case import TestCase
from common.utilities.training import *
class SimpleStateReacher(TestCase):

    def __init__(self, test_str):
        super().__init__(test_str)
        self.number_of_samples = int(self.formula.split(":")[0])
        self.feature_indezes_and_assignments = self.formula.split(":")[1]
        self.f_dict = {}
        for feature_index_and_assignment in self.feature_indezes_and_assignments.split(","):
            feature_index = int(feature_index_and_assignment.split("=")[0])
            assignment = float(feature_index_and_assignment.split("=")[1])
            self.f_dict[feature_index] = assignment
        
        
    
    def check_trajectory(self, trajectory):
        for i in range(len(trajectory)):
            for feature_index in self.f_dict.keys():
                if abs(trajectory[i][0][feature_index] - self.f_dict[feature_index])<=0.1:
                    return False
                # if last i, then also check next_state
                if i == len(trajectory)-1:
                    if abs(trajectory[i][3][feature_index] - self.f_dict[feature_index])<=0.1:
                        return False
        return True



    def run(self, m_project, gimi_env):
        reaches = 0
        for i in range(self.number_of_samples):
            reward, trajectory = execute(m_project, gimi_env, True, 1, printing=False)
            # Check if trajectory 
            if self.check_trajectory(trajectory):
                reaches += 1
        return {"result" : reaches/self.number_of_samples}

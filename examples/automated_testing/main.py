from openai import OpenAI
import sys
from helper import *
import os
import time
import pandas as pd
if __name__ == "__main__":
    env_name = "CartPole-v1"
    model = OpenAI(api_key="") # USE OWN API KEY
    env_paras = extract_environment_parameters(model, env_name, 'system_texts/optimized_env_para_finder.txt')
    #env_paras = ['mode', 'repeat_action_probability', 'difficulty', 'frameskip']
    # para combinations
    print(env_paras)
    combinations = generate_combinations(env_paras)
    print("Combinations:")
    # Print each level of combinations separately
    for level in combinations:
        for combi in level:
            # Convert combit to string
            combi_str = ', '.join(combi)
            print(combi_str)
            para_configs = generate_safety_fuzz_test_para_dict(model, env_name, 'system_texts/system2.txt', combi_str)
            source_code = create_safety_fuzz_test_source_code(env_name, para_configs)
            f = open(f"test_{combi_str}.py", "w")
            f.write(source_code)
            f.close()
            os.system(f"python test_{combi_str}.py")
            exit(0) # For the rest, please pip install gimitest and run the code outside the repository with all of its dependencies.
            # Read tests results
            data_log_file_name = env_name.replace("/", "_") + ".csv"
            df = pd.read_csv(data_log_file_name)
            # Get top 3 best collected_rewards and top 3 worst collected_rewards
            top_3_best = df.nlargest(3, 'collected_reward')
            top_3_worst = df.nsmallest(3, 'collected_reward')
            # Combine the top 3 best and worst
            top_3_best_worst = pd.concat([top_3_best, top_3_worst])
            print(top_3_best_worst)
            analysis_text = safety_test_analysis(model, top_3_best_worst, 'system_texts/safety_test_analyzer.txt')
            print(analysis_text)
            f = open(f"test_{combi_str}_analysis.txt", "w")
            f.write(analysis_text)
            f.close()
            

            break
        break
    # Execute test.py
    
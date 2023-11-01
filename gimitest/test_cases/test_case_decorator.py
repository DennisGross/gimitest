class TestCaseDecorator:

    @staticmethod
    def decorate_test_case_with_test_logger(test_case, test_logger):
        """Internal method to decorate the step function of a gym environment.
        
        Args:
            test_case (object): The test case to decorate.
            test_logger (object): The test result to add.

        Returns:
            object: The decorated test case.
        """
        test_case.step_execute = TestCaseDecorator.__decorate_step_execute_function(test_case, test_case.step_execute, test_logger)
        test_case.episode_execute = TestCaseDecorator.__decorate_episode_execute(test_case, test_case.episode_execute, test_logger)
        return test_case

    @staticmethod
    def __decorate_step_execute_function(test_case, original_step_execute, test_logger):
        """Internal method to decorate the step function of a gym environment.
        
        Args:
            test_case (object): The test case to decorate.
            original_step_execute (function): The original step function.
            test_logger (object): The test result to add.

        Returns:
            function: The decorated step function.
        """
        def wrapper(*action_args, **kwargs):
            current_episode = test_case.episode
            current_step = test_case.steps
            # Call the original step function
            original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info = original_step_execute(*action_args, **kwargs)
            # Store the step
            test_logger.store_episode_step(current_episode, current_step, original_state,  action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info, test_case.meta_data)
            # Increment the step
            test_case.step_increment()
            if original_terminated or original_truncated:
                # Store the episode
                test_logger.store_episode(current_episode, test_case.meta_data)
            return original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info


        return wrapper


    @staticmethod
    def __decorate_episode_execute(test_case, original_episode_execute, test_logger):
        """Internal method to decorate the step function of a gym environment.
        
        Args:
            test_case (object): The test case to decorate.
            original_episode_execute (function): The original episode execute function.
            test_logger (object): The test result to add.

        Returns:
            function: The decorated step function.
        """
        def wrapper(*action_args, **kwargs):
            # Call the original step function
            original_episode_execute(*action_args, **kwargs)
            # Store the episode
            #test_logger.store_episode(test_case.episode, test_case.meta_data)
            # Increment the episode
            test_case.episode_increment()
            # Reset step
            test_case.steps = 0

        return wrapper
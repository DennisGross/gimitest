import gymnasium as gym  # Importing gymnasium as gym to work as the base for the decorator

class EnvDecorator:

    @staticmethod
    def decorate(env, gtest):
        env.tmp_storage_of_state = None
        env.step = EnvDecorator.__decorate_step_function(env, env.step, gtest)
        env.reset = EnvDecorator.__decorate_reset_function(env, env.reset, gtest)
        return env

    @staticmethod
    def __decorate_step_function(env, original_step_function, gtest):
        def wrapper(*action_args, **kwargs):
            gtest.pre_step_configuration()
            try:
                # Try to get agent selection in turn-based games
                agent_selection = env.agent_selection
            except:
                # If not possible, set to None
                agent_selection = None
            
            # Makes it possible to test, for instance, alternative actions
            action = gtest.pre_step_test( agent_selection, action_args[0])
            if action is not None:
                action_args[0] = action

            # Call the original step function
            if agent_selection is None:
                original_next_state, original_reward, original_terminated, original_truncated, original_info = original_step_function(*action_args, **kwargs)
            else:
                original_step_function(*action_args, **kwargs)
                original_next_state, original_reward, original_terminated, original_truncated, original_info = env.last()
                
            gtest.post_step_configuration()
    
            tmp_next_state = original_next_state
            tmp_reward = original_reward
            tmp_terminated = original_terminated
            tmp_truncated = original_truncated
            tmp_info = original_info
            tmp_state, tmp_action_args, tmp_next_state, tmp_reward, tmp_terminated, tmp_truncated, tmp_info = gtest.post_step_test(env.tmp_storage_of_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, tmp_info, agent_selection)
            
            return tmp_next_state, tmp_reward, tmp_terminated, tmp_truncated, tmp_info
            
        
        return wrapper

    @staticmethod
    def __decorate_reset_function(env, original_reset_function, gtest):
        def wrapper(*args, **kwargs):
            gtest.pre_reset_test()
            more_args = gtest.pre_reset_configuration()
            # Check if more_args is instance
            if isinstance(more_args, dict)==False:
                more_args = {}
            # update kwars with more_args
            kwargs.update(more_args)

    
            try:
                next_state, info = original_reset_function(*args, **kwargs)
            except:
                original_reset_function(*args, **kwargs)
                next_state, reward, done, truncated, info = env.last()

            
            tmp_next_state = gtest.post_reset_test()
            if tmp_next_state is not None:
                next_state = tmp_next_state
            
            env.tmp_storage_of_state = next_state

        
            tmp_state = gtest.post_reset_configuration(next_state)
            if tmp_state is not None:
                env.tmp_storage_of_state = tmp_state
            
            return env.tmp_storage_of_state, info
        return wrapper

    
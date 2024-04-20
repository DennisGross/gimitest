class GTestDecorator:

    @staticmethod
    def decorate_with_logger(gtest, glogger):
        gtest.post_step_test = GTestDecorator.__decorate_post_step_test(gtest, gtest.post_step_test, glogger)
        gtest.pre_reset_test = GTestDecorator.__decorate_pre_reset_test(gtest, gtest.pre_reset_test, glogger)
        return gtest
    
    @staticmethod
    def __decorate_post_step_test(gtest, original_post_step_test, glogger):
        def wrapper(*action_args, **kwargs):
            current_episode = gtest.episode
            current_step = gtest.step
            # Try to get agent selection in turn-based games from parameters
            try:
                agent_selection = action_args[-1]   # Last argument is agent_selection
            except:
                # If not possible, set to None
                agent_selection = None
            
            glogger.agent_selection = agent_selection

            # Call the original step function
            original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info = original_post_step_test(*action_args, **kwargs)
            # Store the step
            #print('====================')
            glogger.own_step_storage(current_episode, current_step, original_state,  action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info, gtest.step_data, agent_selection)
            glogger.step_storage(current_episode, current_step, original_state,  action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info, gtest.step_data, agent_selection)
            # Increment the step
            gtest.step_increment()
            if original_terminated or original_truncated:
                # Store the episode
                glogger.own_episode_storage(current_episode, gtest.episode_data, agent_selection)
                glogger.episode_storage(current_episode, gtest.episode_data, agent_selection)

            return original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info

        return wrapper

    @staticmethod
    def __decorate_pre_reset_test(gtest, original_pre_reset_test, glogger):
        def wrapper(*action_args, **kwargs):
            if gtest.episode != -1:
                glogger.own_episode_storage(gtest.episode, gtest.episode_data, glogger.agent_selection)
                glogger.episode_storage(gtest.episode, gtest.episode_data, glogger.agent_selection)
            # Call the original step function
            original_pre_reset_test(*action_args, **kwargs)
            # Increment the episode
            gtest.episode_increment()
        return wrapper
    
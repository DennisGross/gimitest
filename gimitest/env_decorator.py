import gymnasium as gym  # Importing gymnasium as gym to work as the base for the decorator
from PIL import Image
import numpy as np
class EnvDecorator:

    @staticmethod
    def check_return_values(func):
        # Call the function and capture its return value
        result = func()
        # Check if the result is a tuple (multiple return values)
        if isinstance(result, tuple):
            return len(result)
        elif result is None:
            return 0
        else:
            # If it's a single value
            return 1
       

    @staticmethod
    def decorate(env, gtest):
        number_of_return_values = EnvDecorator.check_return_values(env.reset)
        old_style = False
        if number_of_return_values == 1:
            old_style = True
        env.tmp_storage_of_state = None
        env.step = EnvDecorator.__decorate_step_function(env, env.step, gtest, old_style)
        env.reset = EnvDecorator.__decorate_reset_function(env, env.reset, gtest, old_style)
        env.render = EnvDecorator.__decorate_render_function(env, env.render, gtest)
        gtest.env = env
        return env

    @staticmethod
    def __decorate_step_function(env, original_step_function, gtest, old_style):
        def wrapper(*action_args, **kwargs):
            if gtest.decorated:
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
                    action_args = list(action_args)
                    action_args[0] = action
                    action_args = tuple(action_args)

                # Call the original step function
                if agent_selection is None:
                    if old_style:
                        original_next_state, original_reward, original_terminated, original_info = original_step_function(*action_args, **kwargs)
                        original_truncated = False
                    else:
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
                env.tmp_storage_of_state = original_next_state
                return tmp_next_state, tmp_reward, tmp_terminated, tmp_truncated, tmp_info
            else:
                return original_step_function(*action_args, **kwargs)
            
        
        return wrapper

    @staticmethod
    def __decorate_reset_function(env, original_reset_function, gtest, old_style):
        def wrapper(*args, **kwargs):
            if gtest.decorated:
                gtest.pre_reset_test()
                more_args = gtest.pre_reset_configuration()
                # Check if more_args is instance
                if isinstance(more_args, dict)==False:
                    more_args = {}
                # update kwars with more_args
                kwargs.update(more_args)

        
                try:
                    if old_style:
                        next_state = original_reset_function(*args, **kwargs)
                        info = {}
                    else:
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
            else:
                return original_reset_function(*args, **kwargs)
        return wrapper

    @staticmethod
    def __decorate_render_function(env, original_render_function, gtest):
        def wrapper(*args, **kwargs):
            if gtest.decorated:
                # Call the original render function and get the image
                render_result = original_render_function(*args, **kwargs)
                
                if isinstance(render_result, np.ndarray):
                    # If the render result is a numpy array, convert it to a PIL Image
                    gtest.current_image = Image.fromarray(render_result)

                gtest.post_render()

                return render_result
            else:
                return original_render_function(*args, **kwargs)
            
        return wrapper
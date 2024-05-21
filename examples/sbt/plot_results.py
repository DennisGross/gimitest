import sys
import numpy as np

sys.path.append('../utils/')
from logger import Logger
from plotting import plot_inputs_in_a_single_chart, add_inputs_to_chart,\
    add_faults_to_chart, post_process_figure


if __name__ == '__main__':
    import sys
    sys.path.append('../utils/')
    from logger import Logger

    rt_logpath = 'random_testing.txt'
    rt_logs = Logger(rt_logpath).load_logs()
    rt_inputs = np.vstack(rt_logs['input'])
    rt_oracles = rt_logs['oracle'].to_numpy().astype(bool)

    es_logpath = 'es.txt'
    es_logs = Logger(es_logpath).load_logs()
    es_inputs = np.vstack(es_logs['input'])
    es_oracles = es_logs['oracle'].to_numpy().astype(bool)

    fuzzing_logpath = 'fuzzing_logs.txt'
    fuzzing_logs = Logger(fuzzing_logpath).load_logs()
    fuzzing_inputs = np.vstack(fuzzing_logs['input'])
    fuzzing_oracles = fuzzing_logs['oracle'].to_numpy().astype(bool)

    fig, ax = plot_inputs_in_a_single_chart(rt_inputs)
    ax = add_inputs_to_chart(ax, es_inputs)
    ax = add_inputs_to_chart(ax, fuzzing_inputs)

    ax = add_faults_to_chart(ax, rt_inputs, rt_oracles, 'RT', 'xkcd:deep sky blue')
    ax = add_faults_to_chart(ax, es_inputs, es_oracles, 'ST', 'xkcd:fire engine red')
    ax = add_faults_to_chart(ax, fuzzing_inputs, fuzzing_oracles, 'FT', 'xkcd:amber')

    fig, ax = post_process_figure(fig, ax)
    fig.savefig('results_sbt.png')
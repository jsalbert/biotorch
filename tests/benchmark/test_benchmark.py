import os
import subprocess
import pandas as pd
from biotorch.benchmark.run import Benchmark


def test_benchmark(config_bp_path):
    benchmark = Benchmark(config_bp_path)
    benchmark.run()
    current_files = os.listdir('tests/tmp/mnist/le_net/backpropagation_test/')
    expected_files = ['best_acc.txt', 'config.yaml', 'latest_model.pth', 'results.csv', 'results.json',
                      'model_best_acc.pth', 'logs']

    for file in expected_files:
        assert file in current_files


def test_benchmark_command_line_reproducibility_cpu(config_usf_reproducible_path):
    cmd = ["python", "benchmark.py", "--config", config_usf_reproducible_path]
    subprocess.run(cmd)
    results_1 = pd.read_json('tests/tmp/mnist/le_net/usf_test/results.json')
    cmd = ["python", "benchmark.py", "--config", config_usf_reproducible_path]
    subprocess.run(cmd)
    results_2 = pd.read_json('tests/tmp/mnist/le_net/usf_test/results.json')
    pd.testing.assert_frame_equal(results_1, results_2)

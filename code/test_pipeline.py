# filepath: code/test_pipeline.py
"""
自动化smoke测试：确保train/metric/optuna_search主流程可跑通。
"""
import os
import sys
import shutil
import time
import importlib

# 保证能import到code目录下的模块
sys.path.insert(0, os.path.dirname(__file__))

from train import train, config
from metric import run_metric


def test_train_smoke():
    test_config = config.copy()
    test_config['root_dir'] = 'data/sample_data/train'  # 用sample_data做训练
    test_config['epochs'] = 1
    test_config['batch_size'] = 1
    test_config['max_signal'] = 10
    test_config['max_idler'] = 10
    test_config['stack_num'] = 2
    test_config['in_channels'] = (test_config['max_signal'] // test_config['stack_num']) + (test_config['max_idler'] // test_config['stack_num'])
    result = train(test_config)
    assert 'best_val_loss' in result and 'exp_dir' in result
    assert os.path.exists(os.path.join(result['exp_dir'], 'best_model.pth'))
    print("train流程smoke测试通过")
    return result['exp_dir']


def test_metric_smoke(exp_dir):
    # 用刚刚训练的exp_dir，推理时指定sample_data/test
    from metric import run_metric
    run_metric(exp_dir=exp_dir, test_root_dir='data/sample_data/test')
    print("metric流程smoke测试通过")
    # 删除exp_dir
    import shutil
    shutil.rmtree(exp_dir, ignore_errors=True)


def test_optuna_smoke():
    import train
    train.config['epochs'] = 1
    train.config['root_dir'] = 'data/sample_data/train'  # optuna也用sample_data
    import optuna_search
    study = optuna_search.optuna.create_study(direction='minimize')
    study.optimize(optuna_search.objective, n_trials=1)
    print("optuna_search流程smoke测试通过")
    # 删除最新exp目录
    from metric import get_latest_config
    config, exp_dir = get_latest_config('results')
    import shutil
    shutil.rmtree(exp_dir, ignore_errors=True)


if __name__ == "__main__":
    #print("[INFO] smoke test已禁用。上传/CI时请确保data/sample_data已准备好后再启用测试。")
    exp_dir = test_train_smoke()
    test_metric_smoke(exp_dir)
    #test_optuna_smoke()
    print("全部smoke测试通过，测试产物已清理！")

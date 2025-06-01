import optuna 
import torch
from train import train, config
import copy
import os
import time
from metric import run_metric

def objective(trial):
    #采样参数
    trial_config = copy.deepcopy(config)
    trial_config['stack_num'] = trial.suggest_categorical('stack_num', [2, 5, 10, 20])
    trial_config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2)
    trial_config['max_signal'] = trial.suggest_categorical('max_signal', [50, 100, 200])
    trial_config['max_idler'] = trial.suggest_categorical('max_idler', [50, 100, 200])
    trial_config['loss_weight_ssim'] = trial.suggest_float('loss_weight_ssim', 0.5, 2.0)
    trial_config['loss_weight_mse'] = trial.suggest_float('loss_weight_mse', 0.01, 0.5)
    trial_config['loss_weight_perceptual'] = trial.suggest_float('loss_weight_perceptual', 0.01, 0.5)
    #重新计算 in_channels
    trial_config['in_channels'] = (trial_config['max_signal'] // trial_config['stack_num']) + (trial_config['max_idler'] // trial_config['stack_num'])

    #训练模型
    try:
        result = train(trial_config)
        val_loss = result['best_val_loss']
        exp_dir = result['exp_dir']
        # 训练后自动推理，生成预测图片
        run_metric(exp_dir=exp_dir)
    except optuna.TrialPruned:
        raise
    return val_loss

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print(' Best trial:')
    trial = study.best_trial
    print(f' Loss: {trial.value}')
    print(' Params:')
    for key, value in trial.params.items():
        print(f' {key}: {value}')

    # save trials to csv
    df = study.trials_dataframe()
    os.makedirs('results/optuna', exist_ok=True)
    df.to_csv(f'results/optuna/trials{time.strftime("%Y%m%d_%H%M%S")}.csv')

    # save best params
    import json
    with open(f'results/optuna/best_params{time.strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(trial.params, f)
    with open(f'results/optuna/best_value{time.strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
        f.write(f'Best loss: {trial.value}\n')
        f.write(json.dumps(trial.params, indent=2))




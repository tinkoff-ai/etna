import optuna
from etna.auto.optuna.config_sampler import ConfigSampler

def test_dummy():
    
    def objective(trial: optuna.trial.Trial):
        config = {**trial.relative_params, **trial.params}
        x = config["x"]
        return (config["x"] - 2) ** 2
    
    

    study = optuna.create_study(
        sampler=ConfigSampler(configs=[{"x": 1}, {"x": 2}, {"x": 3}])
    )
    study.optimize(objective, n_trials=100)
    assert study.best_params == {"x": 2}
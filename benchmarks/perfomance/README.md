# How to run bench cases  

- Install additional dependencies

```bash
    pip install -r requirements.txt
```
- Override `config.pattern_to_filter`: it should be equal to substring of full path to library: `etna/etna` for example if you just make `git clone`.  
N.B. `etna` is not proper substring cause `venv` folder usually in `..etna/venv` and you'll get noisy results of other libraries  
- Run defined test cases

```bash
python main.py -m pipeline=daily_case1 dataset=daily_20,daily_100,daily_1000,daily_5000
```

## Next steps

- Get high level results with view.ipynb notebook.
- Analyze flamegraph file `speedscope.json` with https://speedscope.app.
- Get insights and start issue if you have idea how to fix performance issues.

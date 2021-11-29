# How to run bench cases  

- Install additional dependencies

```bash
    pip install -r requirements.txt
```

- Run defined test cases

```bash
python main.py -m pipeline=daily_case1 dataset=daily_20,daily_100,daily_1000,daily_5000
```

## Next steps

- Get high level results with view.ipynb notebook.
- Analyze flamegraph file `speedscope.json` with https://speedscope.app.
- Get insights and start issue if you have idea how to fix performance issues.

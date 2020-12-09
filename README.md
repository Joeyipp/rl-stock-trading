## Stock Trading Automation with Deep Reinforcement Learning

### Description

An automated stock trading with Deep Reinforcement Learning (DQN & DDPG) for AAPL, BA, and TSLA with news sentiment and one/ multi-step stock price prediction.

### Flowchart

![Flowchart](https://github.com/Joeyipp/rl-stock-trading/blob/master/images/flowchart_design.png)

### Instructions

1. Install Open MPI from [here](https://stable-baselines.readthedocs.io/en/master/guide/install.html)

> `MacOS: brew install cmake openmpi`

2. Create conda environment

> `conda create --name rl_stock python=3.7`

3. Activate the conda environment

> `conda activate rl_stock`

4. Install requirements

> `pip install -r requirements.txt`

### Run

**Arguments**

- -a, --agent, DQN or DDPG
- -f, --forecast, one or multi
- -s, --sentiment, True or False

**Baseline**

`python run.py -a {DQN|DDPG}`

**Baseline + One/Multi-step Stock Forecast**

`python run.py -a {DQN|DDPG} -f {one|multi}`

**Baseline + News Sentiment Analysis**

`python run.py -a {DQN|DDPG} -s True`

**Baseline + One/Multi-step Stock Forecast + News Sentiment Analysis**

`python run.py -a {DQN|DDPG} -f {one|multi} -s True`

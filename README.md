## Stock Trading Automation with Deep Reinforcement Learning

### Description

An automated stock trading with Deep Reinforcement Learning (DQN & DDPG) for AAPL, BA, and TSLA with news sentiment and one/ multi-step stock price prediction.

For complete report & slide, navigate to `reports`.

### Instructions

To run locally:

1. Clone this repository. `NOTE: Scripts are written with Python 3.7.`

2. Install Open MPI from [here](https://stable-baselines.readthedocs.io/en/master/guide/install.html)

> `MacOS: brew install cmake openmpi`

3. Create conda environment

> `conda create --name rl_stock python=3.7 -y`

4. Activate the conda environment

> `conda activate rl_stock`

5. Install requirements

> `pip install -r requirements.txt`

### Run

**Arguments**

- -a, --agent, DQN or DDPG
- -f, --forecast, one or multi
- -s, --sentiment, True or False

> Baseline

`python run.py -a {DQN|DDPG}`

> Baseline + News Sentiment Analysis

`python run.py -a {DQN|DDPG} -s True`

> Baseline + One/Multi-step Stock Forecast

`python run.py -a {DQN|DDPG} -f {one|multi}`

> Baseline + One/Multi-step Stock Forecast + News Sentiment Analysis

`python run.py -a {DQN|DDPG} -f {one|multi} -s True`

### System Architecture

![Flowchart](https://github.com/Joeyipp/rl-stock-trading/blob/master/images/flowchart_design.png)

### News Sentiment Analysis

![Flowchart](https://github.com/Joeyipp/rl-stock-trading/blob/master/images/sentiment.png)

### Stock Price LSTM Forecast

![Flowchart](https://github.com/Joeyipp/rl-stock-trading/blob/master/images/forecast.png)

### Data Collection

This repo is self-contained with data.
To self-collect for your own stock data, navigate to `utils/collect_data.py` and modify accordingly.

### Credits

- [Practical Deep Reinforcement Learning Approach for Stock Trading](https://arxiv.org/abs/1811.07522)
- [LazyProgrammer](https://github.com/lazyprogrammer)

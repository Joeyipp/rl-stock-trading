import pickle
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
from utils.utils import get_data, make_dir
from environments.multi_stock_env import MultiStockEnv, play_one_episode, get_scaler
from models.dqn import DQNAgent
from models.ddpg import DDPGAgent
from models.lstm_forecast import lstm_forecast
from models.sentiment_analysis import sentiment_analysis

def main():
    # Config
    stock_tickers = ["AAPL", "BA", "TSLA"]
    n_stock = len(stock_tickers)
    n_forecast = 0
    n_sentiment = 0
    models_folder = 'saved_models'
    rewards_folder = 'saved_rewards'
    news_folder = './data/news'
    forecast_window = 10
    num_episodes = 300
    batch_size = 16
    initial_investment = 10000

    # Parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--forecast', type=str, default=None, help='Enable stock forecasting. Select "one" or "multi"')
    parser.add_argument('-s', '--sentiment', type=bool, default=False, help='Enable sentiment analysis. Select "True" or "False"')
    parser.add_argument('-a', '--agent', type=str, default="DQN", help='Select "DQN" or "DDPG"')
    args = parser.parse_args()

    make_dir(models_folder)
    make_dir(rewards_folder)

    # Get data
    data = get_data("./data", stock_tickers)
    print()
    
    # Generate state (features) based on arguments (forecast & sentiment)
    if args.forecast == None and args.sentiment == False:
        data = data.drop(columns="timestamp").iloc[forecast_window:].reset_index(drop=True).values
    
    elif args.forecast != None and args.sentiment == False:
        concat_data = data.iloc[forecast_window:].reset_index(drop=True)
        for ticker in stock_tickers:
            print(f"Performing {ticker} {args.forecast.title()}step Forecast")
            predictions = {}
            predictions[f'{ticker}_Forecast'] = lstm_forecast(models_folder, ticker, data, forecast_window, args.forecast.lower())
    
            # predictions[f'{ticker}_Forecast'] = pd.DataFrame(predictions)
            # predictions[f'{ticker}_Forecast'].index = pd.RangeIndex(forecast_window, forecast_window + len(predictions[f'{ticker}_Forecast']))
            
            concat_data = pd.concat([concat_data, pd.DataFrame(predictions)], join="outer", axis=1)

        print(f"{args.forecast.title()}step Forecasts Added!\n")
        data = concat_data.drop(columns="timestamp").values
        n_forecast = len(stock_tickers)

    elif args.forecast != None and args.sentiment:
        concat_data = data.iloc[forecast_window:].reset_index(drop=True)
        for ticker in stock_tickers:
            print(f"Performing {ticker} {args.forecast.title()}step Forecast")
            predictions = {}
            predictions[f'{ticker}_Forecast'] = lstm_forecast(models_folder, ticker, data, forecast_window, args.forecast.lower())
    
            # predictions[f'{ticker}_Forecast'] = pd.DataFrame(predictions)
            # predictions[f'{ticker}_Forecast'].index = pd.RangeIndex(forecast_window, forecast_window + len(predictions[f'{ticker}_Forecast']))
            
            concat_data = pd.concat([concat_data, pd.DataFrame(predictions)], join="outer", axis=1)

        print(f"{args.forecast.title()}step Forecasts Added!\n")

        for ticker in stock_tickers:
            print(f"Analyzing {ticker} Stock Sentiment")
            sentiment_df = sentiment_analysis(news_folder, ticker)
            
            concat_data = pd.merge(concat_data, sentiment_df, left_on="timestamp", right_on="publishedAt").drop(columns="publishedAt", axis=1)
        
        print("Sentiment Features Added!\n")
        print(concat_data)
        data = concat_data.drop(columns="timestamp").values
        n_forecast = len(stock_tickers)
        n_sentiment = len(stock_tickers)

    elif args.sentiment:
        concat_data = data
        for ticker in stock_tickers:
            print(f"Analyzing {ticker} Stock Sentiment")
            sentiment_df = sentiment_analysis(news_folder, ticker)
            # print(sentiment_df)
            concat_data = pd.merge(concat_data, sentiment_df, left_on="timestamp", right_on="publishedAt").drop(columns="publishedAt", axis=1)
        
        print("Sentiment Features Added!\n")
        data = concat_data.drop(columns="timestamp").values
        n_sentiment = len(stock_tickers)

    n_timesteps, _ = data.shape

    n_train = n_timesteps
    train_data = data[:n_train]
    test_data = data[n_train:]

    # Initialize the MultiStock Environment
    env = MultiStockEnv(train_data, n_stock, n_forecast, n_sentiment, initial_investment, "DQN")
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    # Store the final value of the portfolio (end of episode)
    portfolio_value = []

    ######### DDPG #########
    # Run with DDPG Agent
    if args.agent.lower() == "ddpg":
        env = MultiStockEnv(train_data, n_stock, n_forecast, n_sentiment, initial_investment, "DDPG")
        DDPGAgent(env, num_episodes)
        exit()
    ######### /DDPG #########

    ######### DQN #########
    # Run with DQN
    # play the game num_episodes times
    print("\nRunning DQN Agent...\n")
    for e in range(num_episodes):
        val = play_one_episode(agent, env, scaler, batch_size)
        print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}")
        portfolio_value.append(val) # append episode end portfolio value

    # save the weights when we are done
    # save the DQN
    agent.save(f'{models_folder}/rl/dqn.h5')

    # save the scaler
    with open(f'{models_folder}/rl/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # save portfolio value for each episode
    np.save(f'{rewards_folder}/rl/dqn.npy', portfolio_value)

    print("\nDQN Agent run complete and saved!")

    a = np.load(f'./saved_rewards/rl/dqn.npy')

    print(f"\nCumulative Portfolio Value Average: {a.mean():.2f}, Min: {a.min():.2f}, Max: {a.max():.2f}")
    plt.plot(a)
    plt.title(f"Portfolio Value Per Episode ({args.agent.upper()})")
    plt.ylabel("Portfolio Value")
    plt.xlabel("Episodes")
    plt.show()
    ######### /DQN #########

if __name__ == '__main__':
    main()
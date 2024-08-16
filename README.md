will be filled in later

currently choosing to use alphavantage news and sentiments to make data collection easier than using finbert and manually calculating sentiments

raw - sentiment data and candles data

candles data is used for original model, uses this data to generate 90 day timestep features with 8 features

sentiment data is used for testing later on when doing prediction

merged data is used for finetuning training on sentiment and candles data

interim contains candles feature sets for each stock

interim_merged contains merged feature sets for each stock

processed contains a candles full feature set and a merged full feature set


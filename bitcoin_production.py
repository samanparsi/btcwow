

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

device = torch.device('cpu')

class CNN_LSTM_Expert(nn.Module):
    def __init__(self, in_channels=6):
        super(CNN_LSTM_Expert, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TradingEnv(gym.Env):
    def __init__(self, df, features_cols):
        super().__init__()
        self.df = df
        self.features_cols = features_cols
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(len(features_cols),), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cum_ret = 1.0
        self.history = [1.0]
        return self._get_obs(), {}

    def _get_obs(self):
        return self.df[self.features_cols].iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        leverage = action - 2
        m_ret = self.df['ret'].iloc[self.current_step + 1]
        atr = self.df['atr'].iloc[self.current_step]

        # Risk logic
        sl = max(0.02, atr * 1.5)
        step_ret = (leverage * m_ret) - (abs(leverage) * 0.001)
        if step_ret < -sl: step_ret = -sl

        self.cum_ret *= (1 + step_ret)
        self.history.append(self.cum_ret)

        reward = step_ret * 100
        self.current_step += 1
        done = self.current_step >= len(self.df) - 2 or self.cum_ret < 0.5
        return self._get_obs(), reward, done, False, {}

class TradingSystem:
    def __init__(self, ticker='BTC-USD'):
        self.ticker = ticker
        self.cnn_lstm = CNN_LSTM_Expert().to(device)
        self.rf_expert = RandomForestRegressor(n_estimators=50, max_depth=5)
        self.regime_rf = RandomForestClassifier(n_estimators=50, max_depth=5)
        self.ppo_model = None

    def prepare_data(self):
        df = yf.download(self.ticker, period='5y', interval='1d', progress=False, auto_adjust=True)
        df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        df['ret'] = df['close'].pct_change().shift(-1)
        df['tr'] = np.maximum((df['high'] - df['low']), np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean() / df['close']
        df['rsi'] = (df['close'] - df['close'].rolling(14).min()) / (df['close'].rolling(14).max() - df['close'].rolling(14).min() + 1e-8)
        df['vol_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        df['ma_ratio'] = df['close'] / df['close'].rolling(50).mean()
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_norm'] = (df['obv'] - df['obv'].rolling(20).mean()) / (df['obv'].rolling(20).std() + 1e-8)
        df = df.dropna()
        for col in ['rsi', 'vol_20', 'ma_ratio', 'atr', 'obv_norm']:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        return df

    def train(self, df, seq_len=14):
        split = int(len(df) * 0.8)
        train_df = df.iloc[:split].copy()

        # 1. Train Regime Classifier
        train_df['regime'] = (train_df['vol_20'] > train_df['vol_20'].median()).astype(int)
        self.regime_rf.fit(train_df[['rsi', 'vol_20', 'ma_ratio']], train_df['regime'])

        # 2. Train CNN-LSTM Expert (Fast Epochs)
        feat_cols = ['rsi', 'vol_20', 'ma_ratio', 'regime', 'atr', 'obv_norm']
        X, Y = [], []
        for i in range(len(train_df) - seq_len):
            X.append(train_df[feat_cols].iloc[i:i+seq_len].values.T)
            Y.append(train_df['ret'].iloc[i+seq_len-1])
        X_t, Y_t = torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32).view(-1, 1)
        opt = optim.Adam(self.cnn_lstm.parameters(), lr=0.001)
        for _ in range(50): # Reduced epochs for speed
            opt.zero_grad(); nn.MSELoss()(self.cnn_lstm(X_t), Y_t).backward(); opt.step()

        # 3. Train RF Expert on residuals or raw returns
        self.rf_expert.fit(train_df[feat_cols].iloc[seq_len:], train_df['ret'].iloc[seq_len:])

        # 4. Generate Expert signals for RL
        with torch.no_grad():
            train_df['cnn_sig'] = 0.0
            train_df.iloc[seq_len:, train_df.columns.get_loc('cnn_sig')] = self.cnn_lstm(X_t).flatten().numpy()
        train_df['rf_sig'] = 0.0
        train_df.iloc[seq_len:, train_df.columns.get_loc('rf_sig')] = self.rf_expert.predict(train_df[feat_cols].iloc[seq_len:])

        # 5. Train PPO Decision Maker
        rl_feats = ['cnn_sig', 'rf_sig', 'regime', 'vol_20', 'atr']
        env = DummyVecEnv([lambda: TradingEnv(train_df.iloc[seq_len:], rl_feats)])
        self.ppo_model = PPO('MlpPolicy', env, verbose=0, learning_rate=1e-4)
        self.ppo_model.learn(total_timesteps=50000) # Faster RL training

    def test(self, df, seq_len=14):
        df = df.copy()
        df['regime'] = self.regime_rf.predict(df[['rsi', 'vol_20', 'ma_ratio']])
        feat_cols = ['rsi', 'vol_20', 'ma_ratio', 'regime', 'atr', 'obv_norm']
        X = [df[feat_cols].iloc[i:i+seq_len].values.T for i in range(len(df)-seq_len)]
        with torch.no_grad():
            df['cnn_sig'] = 0.0
            df.iloc[seq_len:, df.columns.get_loc('cnn_sig')] = self.cnn_lstm(torch.tensor(np.array(X), dtype=torch.float32)).flatten().numpy()
        df['rf_sig'] = 0.0
        df.iloc[seq_len:, df.columns.get_loc('rf_sig')] = self.rf_expert.predict(df[feat_cols].iloc[seq_len:])

        env = TradingEnv(df.iloc[seq_len:], ['cnn_sig', 'rf_sig', 'regime', 'vol_20', 'atr'])
        obs, _ = env.reset(); done = False
        while not done:
            action, _ = self.ppo_model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        return env

# Execution
sys = TradingSystem()
df = sys.prepare_data()
sys.train(df)
test_env = sys.test(df.iloc[int(len(df)*0.8):])

print(f"Final ROI: {test_env.cum_ret-1:.2%}")
plt.plot(test_env.history); plt.title('Optimized Hybrid Daily Strategy'); plt.show()
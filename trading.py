import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.callbacks import EarlyStopping
from a3c import A3CAgent
from textblob import TextBlob
from transformers import pipeline
import re
import logging
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import yfinance as yf
from keras.optimizers import Adam, AdamW, RMSprop
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the model and parameter grid
model = RandomForestRegressor()
param_grid = {
    'n_estimators': np.arange(50, 201, 50),
    'max_depth': np.arange(5, 21),
    'min_samples_split': np.arange(2, 11),
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_grid, n_iter=100, cv=5, random_state=42, verbose=2)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Best parameters found: ", best_params)

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)  # or use cross-validation scores
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
print(f"Best Parameters: {best_params}")

# Fetch live data for a stock (e.g., Apple)
stock_data = yf.download('AAPL', period="1d", interval="1m")  # 1-minute interval data

# Use the real-time data in your forecasting model
X_live = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Relevant features
y_live = stock_data['Close']  # Predict next closing price

# Make predictions with your trained model
predictions = model.predict(X_live)


# Sentiment Analysis Module
class SentimentAnalysis:
    # Initialize a transformer-based sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")

    @staticmethod
    def preprocess_text(text):
        """Clean and preprocess text by removing noise like URLs, special characters, etc."""
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
        text = text.lower().strip()  # Convert to lowercase and strip whitespace
        return text

    @staticmethod
    def get_sentiment_blob(text):
        """Sentiment analysis using TextBlob (Polarity: -1 to 1)."""
        return TextBlob(text).sentiment.polarity

    @staticmethod
    def get_sentiment_transformer(text):
        """Sentiment analysis using a transformer-based model."""
        try:
            result = SentimentAnalysis.sentiment_pipeline(text)[0]
            sentiment = result['label']
            score = result['score']
            return sentiment, score
        except Exception as e:
            logging.error(f"Error in transformer-based sentiment analysis: {e}")
            return None, 0

    @staticmethod
    def analyze(text):
        """Preprocess and analyze sentiment using both TextBlob and transformers."""
        text = SentimentAnalysis.preprocess_text(text)
        polarity = SentimentAnalysis.get_sentiment_blob(text)
        transformer_sentiment, transformer_score = SentimentAnalysis.get_sentiment_transformer(text)
        return {
            "polarity": polarity,
            "transformer_sentiment": transformer_sentiment,
            "transformer_score": transformer_score,
        }


# Option Pricing Module
class BlackScholes:
    @staticmethod
    def calculate(S, K, T, r, sigma, option_type="call"):
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "call":
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            elif option_type == "put":
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        except Exception as e:
            logging.error(f"Error in Black-Scholes calculation: {e}")
            return None

# Kelly Criterion Simulation
class KellyCriterion:
    @staticmethod
    def calculate(prob_win, odds):
        """Calculates the optimal bet fraction using the Kelly Criterion."""
        try:
            return max(0, (prob_win * (odds + 1) - 1) / odds)
        except Exception as e:
            logging.error(f"Error in Kelly Criterion calculation: {e}")
            return None

class ForecastingModels:
    @staticmethod
    def dynamic_model_selection(data):
        """Dynamically selects the best forecasting model based on recent performance."""
        models = {
            'ARIMA': ForecastingModels.arima_forecast,
            'LSTM': ForecastingModels.lstm_forecast,
            'SVM': ForecastingModels.svm_forecast,
            'RandomForest': ForecastingModels.random_forest_forecast
        }
        errors = {}
        tscv = TimeSeriesSplit(n_splits=3)

        for model_name, model_func in models.items():
            try:
                error_list = []
                for train_index, test_index in tscv.split(data):
                    train, test = data[train_index], data[test_index]
                    prediction = model_func(train)
                    if prediction is not None:
                        error = mean_squared_error(test, [prediction] * len(test))
                        error_list.append(error)
                if error_list:
                    errors[model_name] = np.mean(error_list)
            except Exception as e:
                logging.error(f"Error evaluating {model_name}: {e}")

        best_model = min(errors, key=errors.get, default=None)
        if best_model is None:
            raise ValueError("No valid model could be selected.")
        logging.info(f"Selected best model: {best_model}")
        return best_model

    @staticmethod
    def arima_forecast(data, order=(5, 1, 0)):
        try:
            model = ARIMA(data, order=order)
            model_fit = model.fit()
            return model_fit.forecast(steps=1)[0]
        except Exception as e:
            logging.error(f"Error in ARIMA forecast: {e}")
            return None

    @staticmethod
    # Modify LSTM to include online learning
    def lstm_forecast(data, look_back=10, epochs=1, batch_size=1):
        """Incremental learning for LSTM with online updates."""
        data = np.array(data)
        X, Y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back])
            Y.append(data[i + look_back])
    
        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
    
        # Update model incrementally
        for i in range(epochs):
            model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=0)
    
        # Predict for the next time step
        last_sequence = data[-look_back:].reshape(1, look_back, 1)
        return model.predict(last_sequence)[0][0]

    @staticmethod
    def svm_forecast(data):
        try:
            X = np.arange(len(data)).reshape(-1, 1)
            Y = np.array(data)
            model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
            model.fit(X, Y)
            return model.predict([[len(data)]])[0]
        except Exception as e:
            logging.error(f"Error in SVM forecast: {e}")
            return None

    @staticmethod
    def random_forest_forecast(data):
        """Use HistGradientBoosting for online learning (tree-based)."""
        X = np.arange(len(data)).reshape(-1, 1)
        Y = np.array(data)
        
        model = HistGradientBoostingRegressor(max_iter=1, warm_start=True)
        model.fit(X, Y)  # Train on available data
        
        # Predict for the next time step
        return model.predict([[len(data)]])[0]


def online_lstm_forecast(data, look_back=10, epochs=1, batch_size=1, model=None):
    """Incremental learning for LSTM with online updates."""
    data = np.array(data)
    X, Y = [], []
    
    # Prepare data with a sliding window approach
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        Y.append(data[i + look_back])
    
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input
    
    # If no model is provided, initialize a new one
    if model is None:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    # Update the model incrementally
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Predict for the next time step
    last_sequence = data[-look_back:].reshape(1, look_back, 1)
    return model.predict(last_sequence)[0][0], model

# Example of calling the function
forecast, updated_model = online_lstm_forecast(data, look_back=10, epochs=1, batch_size=1)

from sklearn.ensemble import HistGradientBoostingRegressor

def online_random_forest_forecast(data):
    """Use HistGradientBoosting for online learning (tree-based)."""
    X = np.arange(len(data)).reshape(-1, 1)
    Y = np.array(data)

    model = HistGradientBoostingRegressor(max_iter=1, warm_start=True)
    model.fit(X, Y)  # Train on available data

    # Predict for the next time step
    return model.predict([[len(data)]])[0]

def feature_importance_based_feature_engineering(data):
    """Generate new features based on RandomForest feature importance."""
    X = np.arange(len(data)).reshape(-1, 1)
    Y = np.array(data)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, Y)

    # Get feature importance and create new features based on them
    importances = model.feature_importances_
    # Example: Create a moving average feature (this could be done dynamically)
    moving_avg = pd.Series(data).rolling(window=5).mean()

    return moving_avg

def adjust_reward_for_risk(reward, position_size, volatility, risk_threshold=0.05):
    """Adjust the reward by penalizing high risk positions."""
    if volatility > risk_threshold:
        reward -= 0.1 * position_size  # Penalize large positions during high volatility
    return reward

# Example usage in the RL agent:
reward = adjust_reward_for_risk(reward, position_size, current_volatility)

from sklearn.linear_model import LogisticRegression

def ensemble_models(models, data):
    """Combine predictions from multiple models using a meta-model."""
    predictions = [model.forecast(data) for model in models]
    meta_model = LogisticRegression()
    meta_model.fit(np.array(predictions).T, data[-len(predictions[0]):])  # Train on the model predictions
    return meta_model.predict(np.array(predictions).T)



# Advanced Hyperparameter Tuning
def advanced_hyperparameter_tuning(model, params, data):
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(model, param_grid=params, cv=TimeSeriesSplit())
    grid_search.fit(data)
    return grid_search.best_params_

class RLTradingAgent:
    def __init__(self, state_size, action_size, optimizer='adam', stop_loss=0.05, max_drawdown=0.2, initial_balance=100000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        
        # Risk management parameters
        self.stop_loss = stop_loss
        self.max_drawdown = max_drawdown
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.drawdown = 0

        # Choose optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(learning_rate=self.learning_rate)
        elif optimizer == 'adamw':
            self.optimizer = AdamW(learning_rate=self.learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop(learning_rate=self.learning_rate)

        # Initialize DQN
        self.dqn = self._build_dqn_agent()
        
        # A3C is more complex and usually involves multiple workers
        # Here, we'll just set up the model for A3C
        self.a3c_model = self._build_a3c_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(1, self.state_size), activation="relu"))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(optimizer=self.optimizer, loss='mse')
        return model

    def _build_dqn_agent(self):
        model = self._build_model()
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = EpsGreedyQPolicy()
        dqn = DQNAgent(model=model, nb_actions=self.action_size, memory=memory, 
                       nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        return dqn

    def _build_a3c_model(self):
        # A3C typically uses a shared network for both actor and critic
        # Here is a simple example; real A3C would involve more complex architecture
        model = Sequential()
        model.add(Dense(64, input_shape=(1, self.state_size), activation="relu"))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(optimizer=self.optimizer, loss='mse')  # Actor typically uses categorical cross-entropy
        return model

    def act(self, state):
        """Choose an action using the DQN agent."""
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        return self.dqn.forward(state)[0]

    def act_a3c(self, state):
        """Choose an action using the A3C model."""
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        action_probabilities = self.a3c_model.predict(state)[0]
        return np.random.choice(self.action_size, p=action_probabilities)

    def remember(self, state, action, reward, next_state, done):
        """Store experiences in the DQN agent's memory."""
        self.dqn.memory.append(state, action, reward, next_state, done)

    def remember_a3c(self, state, action, reward, next_state, done):
        """For A3C, we directly update with each step, but for simplicity:
        This method is a placeholder for how A3C would handle a single trajectory."""
        pass  # A3C doesn't use a memory buffer in the same way

    def replay(self, batch_size):
        """Train the DQN model."""
        self.dqn.fit(batch_size=batch_size, nb_epochs=1, verbose=0)

    def replay_a3c(self):
        """A3C generally doesn't use replay; it updates online."""
        pass  # Placeholder for A3C training logic

    def evaluate_trade(self, current_price, predicted_price):
        # Calculate profit/loss
        profit_loss = current_price - predicted_price
        if profit_loss < -self.stop_loss * current_price:
            print("Stop-loss triggered!")
            return False  # Stop trade

        # Update balance and drawdown
        self.balance += profit_loss
        self.max_balance = max(self.balance, self.max_balance)
        self.drawdown = (self.max_balance - self.balance) / self.max_balance
        
        if self.drawdown > self.max_drawdown:
            print("Maximum drawdown limit reached!")
            return False  # Stop trading

        return True  # Proceed with trade

    def integrate_sentiment(self, state, sentiment_score):
        """Integrate sentiment score into state."""
        return np.append(state, sentiment_score)

# Example usage for DQN:
state = np.random.rand(1, 3)  # Example state
agent = RLTradingAgent(state_size=3, action_size=2)
action = agent.act(state)
print("DQN Action:", action)

# Example usage for A3C:
action_a3c = agent.act_a3c(state)
print("A3C Action:", action_a3c)
 
# RL-enhanced Backtesting
def backtest_with_rl(data, episodes=50, batch_size=32, initial_balance=10000):
    state_size = 3  # Example state size: [price change, SMA, portfolio balance]
    action_size = 3  # Actions: Buy, Sell, Hold
    agent = RLTradingAgent(state_size, action_size)
    
    for episode in range(episodes):
        balance = initial_balance
        positions = 0
        state = np.array([0, 0, balance]).reshape(1, state_size)  # Initial state
        for t in range(len(data) - 1):
            action = agent.act(state)
            next_state = np.array([data[t+1] - data[t], np.mean(data[max(0, t-5):t]), balance]).reshape(1, state_size)
            
            if action == 0:  # Buy
                positions += balance // data[t]
                balance -= positions * data[t] * 1.01  # Transaction cost
            elif action == 1:  # Sell
                balance += positions * data[t] * 0.99  # Transaction cost
                positions = 0
            
            reward = (balance + positions * data[t+1]) - (balance + positions * data[t])
            done = t == (len(data) - 2)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        logging.info(f"Episode {episode + 1}/{episodes} - Final Balance: ${balance + positions * data[-1]:.2f}")
    
    return balance + positions * data[-1]


# Sharpe Ratio Calculation
def calculate_sharpe(balance_history, risk_free_rate=0.02):
    returns = np.diff(balance_history) / balance_history[:-1]
    return (np.mean(returns) - risk_free_rate) / np.std(returns)

# Monte Carlo Simulation Module
def monte_carlo_simulation(current_price, n_simulations=1000, n_days=252, volatility=None):
    """Simulate future price paths based on historical volatility or provided value."""
    try:
        if volatility is None:
            returns = np.diff(np.log(data.values))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        simulated_paths = []
        for _ in range(n_simulations):
            daily_returns = np.random.normal(loc=0, scale=volatility / np.sqrt(252), size=n_days)
            price_path = [current_price]
            for ret in daily_returns:
                price_path.append(price_path[-1] * np.exp(ret))
            simulated_paths.append(price_path)
        return simulated_paths
    except Exception as e:
        logging.error(f"Error in Monte Carlo simulation: {e}")
        return []


# Main Workflow
if __name__ == "__main__":
    # Fetch Historical Data
    ticker = "AAPL"
    try:
        data = yf.download(ticker, interval="1d", period="1y")['Close']
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        exit()

    # Sentiment Analysis
    news_headline = "Apple announces record-breaking earnings this quarter."
    sentiment_score = SentimentAnalysis.get_sentiment(news_headline)

    # Monte Carlo Simulation
    current_price = data.values[-1]
    monte_carlo_results = monte_carlo_simulation(current_price)
    
    # Interactive Plot
    fig = go.Figure()
    for path in monte_carlo_results:
        fig.add_trace(go.Scatter(y=path, mode='lines', line=dict(color='blue', width=0.5), opacity=0.1))
    fig.add_trace(go.Scatter(y=[current_price], mode='markers', marker=dict(color='red', size=8), name='Current Price'))
    fig.update_layout(title="Monte Carlo Simulation of Future Price Paths", xaxis_title="Days", yaxis_title="Price")
    fig.show()

    # Dynamic Model Selection
    best_model = ForecastingModels.dynamic_model_selection(data.values)
    logging.info(f"Best model selected: {best_model}")

    # Backtesting with Metrics
    final_balance, balance_history = backtest_strategy(data.values, generate_trade_signal)
    sharpe_ratio = calculate_sharpe(balance_history)
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

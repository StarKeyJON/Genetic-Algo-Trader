import time
import requests
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Define the genetic algorithm parameters
mutation_rate = 0.1  # Mutation rate

# Define the trading strategy
class TradingStrategy:
    def __init__(self, window=20, ma_window=21, rsi_window=14, ema_window=10, atr_window=5, threshold=0.5, n_estimators=100, max_depth=7):
        self._window = window
        self._ma_window = ma_window
        self._rsi_window = rsi_window
        self._ema_window = ema_window
        self._atr_window = atr_window
        self._threshold = threshold
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self.model = None

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, value):
        self._window = value
    
    @property
    def ma_window(self):
        return self._ma_window

    @ma_window.setter
    def ma_window(self, value):
        self._ma_window = value

    @property
    def rsi_window(self):
        return self._rsi_window

    @rsi_window.setter
    def rsi_window(self, value):
        self._rsi_window = value

    @property
    def ema_window(self):
        return self._ema_window

    @ema_window.setter
    def ema_window(self, value):
        self._ema_window = value

    @property
    def atr_window(self):
        return self._atr_window

    @atr_window.setter
    def atr_window(self, value):
        self._atr_window = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    @property
    def n_estimators(self):
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, value):
        self._n_estimators = value

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, value):
        self._max_depth = value
    
    def trade(self, data):
        """The trading function."""
        if len(data) < self.window:
            return np.zeros(len(data))

        # window = self.window
        data.loc[:, 'returns'] = data['price'].pct_change()
        data.loc[:, 'volatility'] = data['returns'].rolling(window=int(self.window)).std()
        data.loc[:, 'ma'] = data['price'].rolling(window=int(self.window)).mean()
        data.loc[:, 'rsi'] = self.rsi(data['price'], n=int(self.rsi_window))

        # Add new features
        data.loc[:, 'ema'] = data['price'].ewm(span=int(self.ema_window), adjust=False).mean()
        data.loc[:, 'atr'] = self.atr(data, n=self.atr_window)
        # print(data)
        data.dropna(inplace=True)

        # Generate the labels
        data.loc[:, 'label'] = np.where(data['price'].shift(-1) > data['price'], 1, -1)

        # Fit the model
        self.model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth))
        X = data[['returns', 'volatility', 'ma', 'rsi', 'ema', 'atr']]
        y = data['label']
        # print(X, y)
        self.model.fit(X, y)

        data.loc[:, 'prediction'] = self.model.predict(X)
        data.loc[:, 'strategy_returns'] = self.model.predict_proba(X)[:,0] * data['returns']
        print(data)
        # Calculate the final P&L
        pnl = data.iloc[self.window:]['strategy_returns'].cumsum()
        if pnl.values.any(): print(pnl.values[0])
        return pnl.values


    def atr(self, data, n):
        high = data['price'].shift(1).rolling(window=n, min_periods=n).max()
        low = data['price'].shift(1).rolling(window=n, min_periods=n).min()
        tr1 = abs(high - low)
        tr2 = abs(high - data['price'].shift(1))
        tr3 = abs(low - data['price'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=n).mean()
        return atr

        
    def rsi(self, prices, n=14):
        deltas = np.diff(prices)
        seed = deltas[:n+1]
        up = seed[seed >= 0].sum()/n
        down = -seed[seed < 0].sum()/n
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:n] = 100. - 100./(1.+rs)
        
        for i in range(n, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up*(n-1) + upval)/n
            down = (down*(n-1) + downval)/n
            rs = up/down
            rsi[i] = 100. - 100./(1.+rs)
        
        return rsi
        
    def save(self, filename):
        """Save the trading strategy to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load the trading strategy from a file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

# Define the fitness function
def fitness_function(individual, data):
    """The fitness function."""
    if not isinstance(individual, TradingStrategy):
        if isinstance(individual, int) or isinstance(individual, float):
            return 0
        if len(individual) == 9:
            individual = TradingStrategy(*individual)
        else:
            return 0

    if len(data) < individual.window + 20:
        return 0
    pnl = individual.trade(data)
    if len(pnl) == 0:
        return 0  # or some other default fitness score
    if pnl[-1] < 0:
        return 0  # or some other fitness score for negative P&L
    return sum(pnl)  # or some other fitness score based on P&L

# Define the population initialization function
def init_population(poop_size):
    """Initialize the population."""
    population = []
    for i in range(poop_size):
        individual = [random.randint(5, 50), random.randint(3, 30), random.randint(7, 21), random.randint(10, 21), random.randint(5, 11), random.uniform(0, 1), random.randint(1, 500), random.randint(1, 10)]
        population.append(individual)
    return population

# Define the selection function
def selection(population, fitness_scores, num_to_select=2, num_elites=1):
    # Ensure that there are at least as many non-zero fitness scores as the number of individuals to select
    non_zero_idxs = np.flatnonzero(fitness_scores)
    if len(non_zero_idxs) < num_to_select:
        non_zero_idxs = np.arange(len(population))

    # Select the elites
    elite_idxs = np.argsort(fitness_scores)[::-1][:num_elites]
    elites = [population[i] for i in elite_idxs]

    # Perform tournament selection to select the remaining individuals
    idxs = []
    while len(idxs) < num_to_select - num_elites:
        tournament_idxs = np.random.choice(non_zero_idxs, size=2, replace=False)
        tournament_fitness_scores = [fitness_scores[i] for i in tournament_idxs]
        winner_idx = tournament_idxs[np.argmax(tournament_fitness_scores)]
        idxs.append(winner_idx)

    return elites + [population[idx] for idx in idxs]


def crossover(parent1, parent2):
    if not isinstance(parent1, TradingStrategy):
        parent1 = TradingStrategy(*parent1)
    if not isinstance(parent2, TradingStrategy):
        parent2 = TradingStrategy(*parent2)

    """Crossover the parents."""
    child1 = TradingStrategy(window=parent1.window, ma_window=parent1.ma_window, rsi_window=parent1.rsi_window, ema_window=parent1.ema_window, atr_window=parent1.atr_window, threshold=parent1.threshold,
                            n_estimators=parent1.n_estimators, max_depth=parent1.max_depth)
    child2 = TradingStrategy(window=parent2.window, ma_window=parent2.ma_window, rsi_window=parent2.rsi_window, ema_window=parent2.ema_window, atr_window=parent2.atr_window, threshold=parent2.threshold,
                            n_estimators=parent2.n_estimators, max_depth=parent2.max_depth)
    if random.random() < 0.5:
        child1.window = parent2.window
    else:
        child1.window = parent1.window
    if random.random() < 0.5:
        child1.ma_window = parent2.ma_window
    else:
        child1.ma_window = parent1.ma_window

    if random.random() < 0.5:
        child1.rsi_window = parent2.rsi_window
    else:
        child1.rsi_window = parent1.rsi_window
    if random.random() < 0.5:
        child1.ema_window = parent2.ema_window
    else:
        child1.ema_window = parent1.ema_window
    if random.random() < 0.5:
        child1.atr_window = parent2.atr_window
    else:
        child1.atr_window = parent1.atr_window
    if random.random() < 0.5:
        child1.threshold = parent2.threshold
    else:
        child1.threshold = parent1.threshold
    if random.random() < 0.5:
        child1.n_estimators = parent2.n_estimators
    else:
        child1.n_estimators = parent1.n_estimators
    if random.random() < 0.5:
        child1.max_depth = parent2.max_depth
    else:
        child1.max_depth = parent1.max_depth
        
    if random.random() < 0.5:
        child2.window = parent1.window
    else:
        child2.window = parent2.window

    if random.random() < 0.5:
        child2.ma_window = parent1.ma_window
    else:
        child2.ma_window = parent2.ma_window

    if random.random() < 0.5:
        child2.rsi_window = parent1.rsi_window
    else:
        child2.rsi_window = parent2.rsi_window
    if random.random() < 0.5:
        child2.ema_window = parent1.ema_window
    else:
        child2.ema_window = parent2.ema_window
    if random.random() < 0.5:
        child2.atr_window = parent1.atr_window
    else:
        child2.atr_window = parent2.atr_window

    if random.random() < 0.5:
        child2.threshold = parent1.threshold
    else:
        child2.threshold = parent2.threshold
    if random.random() < 0.5:
        child2.n_estimators = parent1.n_estimators
    else:
        child2.n_estimators = parent2.n_estimators
    if random.random() < 0.5:
        child2.max_depth = parent1.max_depth
    else:
        child2.max_depth = parent2.max_depth
    child1.model = None
    child2.model = None
    return child1, child2

def mutation(individual):
    # """Mutate the individual."""
    if isinstance(individual, TradingStrategy):
        # Create a new TradingStrategy object with some properties randomly mutated while keeping others unchanged
        new_individual = TradingStrategy(window=individual.window, ma_window=individual.ma_window,
                                         rsi_window=individual.rsi_window, ema_window=individual.ema_window,
                                         atr_window=individual.atr_window, threshold=individual.threshold,
                                         n_estimators=individual.n_estimators, max_depth=individual.max_depth)
        if random.random() < mutation_rate:
            new_individual.window = max(5, min(50, int(individual.window) + random.randint(-1, 1)))

        if random.random() < mutation_rate:
            new_individual.ma_window = max(3, min(30, int(individual.ma_window) + random.randint(-1, 1)))

        if random.random() < mutation_rate:
            new_individual.rsi_window = max(7, min(28, int(individual.rsi_window) + random.randint(-1, 1)))

        if random.random() < mutation_rate:
            new_individual.ema_window = max(5, min(20, int(individual.ema_window) + random.randint(-1, 1)))

        if random.random() < mutation_rate:
            new_individual.atr_window = max(3, min(11, int(individual.atr_window) + random.randint(-1, 1)))

        if random.random() < mutation_rate:
            new_individual.threshold = max(0, min(1, individual.threshold + random.uniform(-0.1, 0.1)))

        if random.random() < mutation_rate:
            new_individual.n_estimators = max(1, min(500, int(individual.n_estimators) + random.randint(-10, 10)))

        if random.random() < mutation_rate:
            new_individual.max_depth = max(1, min(10, int(individual.max_depth) + random.randint(-2, 2)))

        new_individual.model = None
        return new_individual

    # If individual is not a TradingStrategy object, mutate each parameter
    new_window = max(5, min(50, int(individual[0]) + random.randint(-1, 1)))

    new_ma_window = max(3, min(30, int(individual[1]) + random.randint(-1, 1)))

    new_rsi_window = max(7, min(28, int(individual[2]) + random.randint(-1, 1)))

    new_ema_window = max(5, min(20, int(individual[3]) + random.randint(-1, 1)))

    new_atr_window = max(3, min(11, int(individual[4]) + random.randint(-1, 1)))

    new_threshold = max(0, min(1, individual[5] + random.uniform(-0.1, 0.1)))

    new_n_estimators = max(1, min(500, int(individual[6]) + random.randint(-10, 10)))

    new_max_depth = max(1, min(10, int(individual[7]) + random.randint(-2, 2)))

    return [new_window, new_ma_window, new_rsi_window, new_ema_window, new_atr_window, new_threshold, new_n_estimators, new_max_depth]


# Define the main function for running the genetic algorithm
def run_genetic_algorithm(population_size, num_generations, data):
    # Create initial population
    population = init_population(poop_size=population_size)

    # Evaluate fitness of the initial population
    fitness_scores = [fitness_function(individual, data) for individual in population]
    # print("Fitness scores: ", fitness_scores)

    for generation in range(num_generations):
        # Select parents for breeding
        parents = [selection(population, fitness_scores)[1:] for i in range(population_size)]

        # Create new generation through crossover and mutation
        offspring = []
        # print("Parents list: ", parents)
        for i in range(population_size // 2):
            parent1, parent2 = random.sample(parents, 2)
            offspring1, offspring2 = crossover(parent1[0], parent2[0])
            offspring1 = mutation(offspring1)
            offspring2 = mutation(offspring2)
            offspring.append(offspring1)
            offspring.append(offspring2)

        # Evaluate the fitness of the offspring
        offspring_fitness = [fitness_function(offspring[i], data) for i in range(len(offspring))]

        # Combine the parents and offspring
        combined_population = population + offspring
        combined_fitness_scores = fitness_scores + offspring_fitness

        # Select the next generation
        population = selection(combined_population, combined_fitness_scores, num_to_select=population_size)

        # Update the fitness scores
        fitness_scores = [fitness_function(individual, data) for individual in population]
        print(fitness_scores)
        # Print the fitness score of the best individual in each generation
        best_fitness = max(fitness_scores)
        print(f"Generation {generation + 1}: Best fitness = {best_fitness}")

    # Return the best individual
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual


# Define the main function
def main():
    # Fetch historical data from the CoinGecko API
    crypto_id = 'ethereum'
    vs_currency = 'usd'
    days = 60
    api_url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency={vs_currency}&days={days}'
    response = requests.get(api_url)
    data = response.json()
    prices = [item[1] for item in data['prices']]
    timestamps = [item[0] for item in data['prices']]
    df = pd.DataFrame({'timestamp': timestamps, 'price': prices})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.to_csv("timestamps.csv", index=False)
    # print(df)

    # Run the genetic algorithm to generate a trading strategy
    best_individual = run_genetic_algorithm(population_size=5000, num_generations=20, data=df)
    # print(best_individual) # Print the trading strategy of the best individual

    # Save the best individual as a pickle file
    best_individual.save('./gen3_pickle/best_trading_strategy.pkl')

   

    # Load the best individual and use it for trading
    best_individual = TradingStrategy.load('./gen3_pickle/best_trading_strategy.pkl')
    pnl = best_individual.trade(df)
    print(pnl) # Print the P&L of the trading strategy on the historical data

 # Save the best individual as a CSV file
    best_individual_df = pd.DataFrame({'pNl': [pnl],
                                       'window': [best_individual.window],
                                       'ma_window': [best_individual.ma_window],
                                       'rsi_window': [best_individual.rsi_window],
                                       'ema_window': [best_individual.ema_window],
                                       'atr_window': [best_individual.atr_window],
                                        'threshold': [best_individual.threshold],
                                        'n_estimators': [best_individual.n_estimators],
                                        'max_depth': [best_individual.max_depth]})
    best_individual_df.to_csv(f'./gen3_res/{crypto_id}_best_trading_strategy_{days}_days{int(time.time())}.csv', index=False)

    # Evaluate the trading strategy on out-of-sample data
    sample_days = 90
    api_url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency={vs_currency}&days={sample_days}'
    response = requests.get(api_url)
    data = response.json()
    prices = [item[1] for item in data['prices']]
    timestamps = [item[0] for item in data['prices']]
    df_out_of_sample = pd.DataFrame({'timestamp': timestamps, 'price': prices})
    df_out_of_sample['timestamp'] = pd.to_datetime(df_out_of_sample['timestamp'], unit='ms')
    pnl_out_of_sample = best_individual.trade(df_out_of_sample.tail(sample_days))
    print("Trade sample ", pnl_out_of_sample, " Best Individual: ", best_individual.ma_window, best_individual.rsi_window, best_individual.ema_window, best_individual.atr_window) # Print the P&L of the trading strategy on the out-of-sample data


if __name__ == '__main__':
    main()

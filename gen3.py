import requests
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Define the genetic algorithm parameters
pop_size = 200  # Population size
num_generations = 100  # Number of generations to run
mutation_rate = 0.1  # Mutation rate

# Define the trading strategy
class TradingStrategy:
    def __init__(self, window=50, threshold=0.5, n_estimators=100, max_depth=5):
        self._window = window
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
        if len(data) < self.window + 1:
            return np.zeros(len(data))

        # window = self.window
        data['returns'] = data['price'].pct_change()
        data['volatility'] = data['returns'].rolling(window=int(self.window)).std()
        data['ma'] = data['price'].rolling(window=int(self.window)).mean()
        data['rsi'] = self.rsi(data['price'], n=int(self.window))

        # Add new features
        data['ema'] = data['price'].ewm(span=int(self.window), adjust=False).mean()
        data['atr'] = self.atr(data, n=int(self.window))
        print(data)
        data.dropna(inplace=True)

        # Generate the labels
        data['label'] = np.where(data['price'].shift(-1) > data['price'], 1, -1)

        # Fit the model
        self.model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth))
        X = data[['returns', 'volatility', 'ma', 'rsi', 'ema', 'atr']]
        y = data['label']
        self.model.fit(X, y)

        data.loc[:, 'prediction'] = self.model.predict(X)
        data.loc[:, 'strategy_returns'] = self.model.predict_proba(X)[:,1] * data['returns']

        # Calculate the final P&L
        pnl = data.iloc[self.window:]['strategy_returns'].cumsum()

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
    if isinstance(individual, TradingStrategy):
        trading_strategy = individual
    else:
        trading_strategy = TradingStrategy(*individual)

    pnl = trading_strategy.trade(data)
    if len(pnl) == 0:
        return 0  # or some other default fitness score
    if pnl[-1] < 0:
        return 0  # or some other fitness score for negative P&L
    return sum(pnl)  # or some other fitness score based on P&L

# Define the population initialization function
def init_population():
    """Initialize the population."""
    population = []
    for i in range(pop_size):
        individual = [random.randint(5, 50), random.uniform(0, 1), random.randint(1, 500), random.randint(1, 10)]
        population.append(individual)
    return population

# Define the selection function
def selection(population, fitness_scores, num_to_select=2, num_elites=1):
    # Calculate probability distribution based on fitness scores
    fitness_sum = sum(fitness_scores)
    if fitness_sum == 0:
        probs = np.ones(len(fitness_scores)) / len(fitness_scores)
    else:
        probs = np.array(fitness_scores) / fitness_sum

    # Ensure that there are at least as many non-zero probabilities as the number of individuals to select
    num_nonzero_probs = np.count_nonzero(probs)
    if num_nonzero_probs < num_to_select:
        idxs = np.random.choice(len(population), size=num_to_select - num_elites, replace=True)
    else:
        idxs = np.random.choice(len(population), size=num_to_select - num_elites, replace=False, p=probs)

    # Select the elites
    elite_idxs = np.argsort(fitness_scores)[::-1][:num_elites]
    elites = [population[i] for i in elite_idxs]

    return elites + [population[idx] for idx in idxs]

def crossover(parent1, parent2):
    parent1 = TradingStrategy(*parent1)
    parent2 = TradingStrategy(*parent2)
    """Crossover the parents."""
    child1 = TradingStrategy(window=parent1.window, threshold=parent1.threshold,
                            n_estimators=parent1.n_estimators, max_depth=parent1.max_depth)
    child2 = TradingStrategy(window=parent2.window, threshold=parent2.threshold,
                            n_estimators=parent2.n_estimators, max_depth=parent2.max_depth)
    if random.random() < 0.5:
        child1.window = parent2.window
    else:
        child1.window = parent1.window
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
    """Mutate the individual."""
    if isinstance(individual, TradingStrategy):
        return TradingStrategy()
    # print(individual.window)
    new_window = max(5, min(50, int(individual[0]) + random.randint(-1, 1)))

    new_threshold = max(0, min(1, individual[1] + random.uniform(-0.1, 0.1)))

    new_n_estimators = max(1, min(500, int(individual[2]) + random.randint(-10, 10)))

    new_max_depth = max(1, min(10, int(individual[3]) + random.randint(-2, 2)))

    return [new_window, new_threshold, new_n_estimators, new_max_depth]


# Define the main function for running the genetic algorithm
def run_genetic_algorithm(population_size, num_generations, data):
    # Create initial population
    population = init_population()

    # Evaluate fitness of the initial population
    fitness_scores = [fitness_function(individual, data) for individual in population]

    for generation in range(num_generations):
        # Select parents for breeding
        parents = [selection(population, fitness_scores, num_elites=1) for i in range(population_size)]

        # Create new generation through crossover and mutation
        offspring = []
        for i in range(population_size // 2):
            parent1, parent2 = random.sample(parents, 2)
            offspring1, offspring2 = crossover(parent1, parent2)
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
    days = 90
    api_url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency={vs_currency}&days={days}'
    response = requests.get(api_url)
    data = response.json()
    prices = [item[1] for item in data['prices']]
    timestamps = [item[0] for item in data['prices']]
    df = pd.DataFrame({'timestamp': timestamps, 'price': prices})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    print(df)

    # Run the genetic algorithm to generate a trading strategy
    best_individual = run_genetic_algorithm(population_size=20, num_generations=1000, data=df)
    print(best_individual) # Print the trading strategy of the best individual

    # Save the best individual as a pickle file
    best_individual.save('best_trading_strategy.pkl')

   

    # Load the best individual and use it for trading
    best_individual = TradingStrategy.load('best_trading_strategy.pkl')
    pnl = best_individual.trade(df)
    print(pnl) # Print the P&L of the trading strategy on the historical data

 # Save the best individual as a CSV file
    best_individual_df = pd.DataFrame({'pNl': [pnl],
                                       'window': [best_individual.window],
                                        'threshold': [best_individual.threshold],
                                        'n_estimators': [best_individual.n_estimators],
                                        'max_depth': [best_individual.max_depth]})
    best_individual_df.to_csv('best_trading_strategy.csv', index=False)

    # Evaluate the trading strategy on out-of-sample data
    days_out_of_sample = 30
    api_url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency={vs_currency}&days={days+days_out_of_sample}'
    response = requests.get(api_url)
    data = response.json()
    prices = [item[1] for item in data['prices']]
    timestamps = [item[0] for item in data['prices']]
    df_out_of_sample = pd.DataFrame({'timestamp': timestamps, 'price': prices})
    df_out_of_sample['timestamp'] = pd.to_datetime(df_out_of_sample['timestamp'], unit='ms')
    pnl_out_of_sample = best_individual.trade(df_out_of_sample.tail(days_out_of_sample))
    print("Trade sample ", pnl_out_of_sample) # Print the P&L of the trading strategy on the out-of-sample data


if __name__ == '__main__':
    main()

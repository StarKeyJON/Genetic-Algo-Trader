# Genetic Algorithm Trading Strategy

This is an experimental project to implement a genetic algorithm to generate a trading strategy. The genetic algorithm works by iteratively generating a population of candidate solutions (in this case, trading strategies), evaluating their fitness, and breeding the fittest individuals to create the next generation of solutions. This process continues until a satisfactory solution is found or a maximum number of generations is reached.

The genetic algorithm works by creating a population of random individuals, each representing a set of values for the trading strategy parameters. The fitness of each individual is then evaluated based on its performance on historical data, and the individuals with the highest fitness scores are selected to breed and create the next generation of individuals. The process is repeated for a set number of generations, with the aim of finding the individual with the highest fitness score, which represents the optimal trading strategy.

The genetic algorithm is a powerful optimization technique that is widely used in machine learning and other fields. It is based on the principle of natural selection and genetic variation, and has been shown to be effective in a wide range of applications.

## Overview of the Code

The code is written in Python and uses several libraries, including requests, pandas, numpy, and scikit-learn. It defines a TradingStrategy class, which represents a candidate solution (i.e., a trading strategy), and a FitnessFunction function, which evaluates the fitness of a given trading strategy on historical price data. The code then uses a genetic algorithm to generate a population of trading strategies, evaluate their fitness, and breed the fittest individuals to create the next generation of solutions.

The code also uses several techniques for preprocessing and feature engineering the historical price data. These include:

Rolling windows: a technique for calculating moving averages and other rolling statistics on the historical price data. Rolling windows are used to calculate the volatility, moving average, and other features of the price data.
Relative Strength Index (RSI): a technical indicator that measures the strength of a security by comparing its average gains to its average losses over a given period. RSI is used to calculate the RSI feature of the price data.
Average True Range (ATR): a technical indicator that measures volatility by taking into account any gaps in the price movement. ATR is used to calculate the ATR feature of the price data.
Exponential Moving Average (EMA): a type of moving average that gives more weight to recent prices. EMA is used to calculate the EMA feature of the price data.

The random forest classifier is used in the trading strategy to predict the direction of the next day's price movement based on a set of features calculated from historical price data. The set of features used includes returns, volatility, moving averages, relative strength index (RSI), exponential moving average (EMA), and average true range (ATR). These features are calculated for a given window size and are used to train the random forest classifier.

The random forest classifier is an ensemble learning algorithm that combines multiple decision trees to improve the accuracy and stability of the predictions. It works by building a large number of decision trees, each of which is trained on a different random subset of the training data and a different random subset of the features. During prediction, the random forest classifier aggregates the predictions of all the decision trees to produce a final prediction.

In the trading strategy code, the random forest classifier is implemented using the scikit-learn library. The classifier is first trained on the historical price data using the features described above, and the resulting model is used to predict the direction of the next day's price movement. The predicted direction is used to generate trading signals, which are used to calculate the profit and loss (P&L) of the trading strategy.

It's worth noting that the choice of classifier and its hyperparameters can significantly impact the performance of the trading strategy. In this code, the hyperparameters of the random forest classifier (i.e., the number of trees and the maximum depth of each tree) are optimized using a genetic algorithm to find the combination that produces the highest fitness score. However, the effectiveness of this approach may depend on the specific trading problem and dataset. Therefore, it's important to carefully evaluate the performance of the classifier and experiment with different hyperparameters to find the optimal combination.

## Explanation of the Code

### Libraries

The code starts by importing the required libraries, including requests, pandas, numpy, random, pickle, sklearn.ensemble.RandomForestClassifier, sklearn.preprocessing.StandardScaler, and sklearn.pipeline.make_pipeline. These libraries are used throughout the code to fetch and preprocess historical price data, create and evaluate trading strategies, and implement the genetic algorithm.

### Genetic Algorithm Parameters

Next, the code defines several parameters that are used by the genetic algorithm, including the population size, number of generations to run, and mutation rate. These parameters control the behavior of the genetic algorithm and can be adjusted to improve its performance.

### Trading Strategy Class

The TradingStrategy class represents a candidate solution (i.e., a trading strategy) and contains several properties that define its behavior, including the window size, threshold, number of estimators, and maximum depth. The class also contains methods for trading on historical price data, saving and loading the strategy to and from a file, and calculating technical indicators such as the average true range (ATR) and relative strength index (RSI).

### Fitness Function

The fitness_function function evaluates the fitness of a given trading strategy on historical price data. It uses the TradingStrategy class to trade on the data and calculates the total profit and loss (P&L) of the strategy. The fitness score is calculated based on the P&L, with higher P&L resulting in higher fitness scores.

### Population Initialization Function

The init_population function initializes the initial population of trading strategies. It creates a list of pop_size individuals, each with randomly selected values for the window size, threshold, number of estimators, and maximum depth. These individuals are used as the starting point for the genetic algorithm.

### Selection Function

The selection function selects individuals from the current population to breed and create the next generation of solutions. It uses a fitness-based selection algorithm to select the fittest individuals, with a bias towards the fittest individuals (i.e., elites). The function also ensures that the selected individuals have non-zero probabilities of being selected, to avoid division by zero errors.

### Crossover Function

The crossover function creates offspring individuals by breeding two parent individuals. It randomly selects traits from each parent to create two new individuals.

### Mutation Function

The mutation function mutates a given individual by randomly adjusting its properties within a certain range. This introduces new genetic material into the population and helps to explore new areas of the solution space.

Overall, the code represents an experimental implementation of the genetic algorithm for generating a trading strategy. While the algorithm has been shown to be effective in many applications, it is important to note that the effectiveness of the algorithm in this specific application is uncertain and may depend on a variety of factors, including the quality and quantity of historical data used for training, the choice of parameters and features used in the trading strategy, and market conditions during the period in which the strategy is deployed. Therefore, the results obtained from this implementation should be interpreted with caution and should not be used as the sole basis for making investment decisions.

It is worth noting that genetic algorithms have been applied to a wide range of optimization problems in various fields, including finance and economics. The use of genetic algorithms in trading strategy generation has been studied extensively in the literature, with many successful applications reported (e.g., [1], [2], [3]). However, it is important to note that genetic algorithms are not a silver bullet and may not always lead to the best possible solutions [4]. As with any optimization method, the effectiveness of genetic algorithms depends on the problem being solved, the quality of the input data, and the choice of algorithm parameters.

References:
[1] X. Wang, Q. Sun, and X. Wang, “Designing a financial trading system using a multi-objective optimization strategy based on genetic algorithms,” Journal of Systems Science and Complexity, vol. 28, no. 5, pp. 1165–1179, 2015.
[2] J. E. Molina, A. C. B. Delgado, and R. C. Fernandez, “Using genetic algorithms to build technical trading rules based on multiple technical indicators,” Expert Systems with Applications, vol. 38, no. 10, pp. 12346–12353, 2011.
[3] H. Wang and L. Li, “A trading strategy based on the simple moving average and optimized by genetic algorithm,” in Proceedings of the 3rd International Conference on Computer Engineering and Technology, 2011, pp. 276–280.
[4] J. H. Holland, Adaptation in natural and artificial systems. MIT Press, 1992.

## License

This script is licensed under the GNU General Public License v3.0. See LICENSE for more information.

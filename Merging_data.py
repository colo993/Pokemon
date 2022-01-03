# In this part I merge to data sets 'Pokemon' and 'Combat'.
# I created two different data sets, one for Machine learning purpose where all columns have this same size
# Next data set is for visualization and further data exploration. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pokemon = pd.read_csv('C:/Users/Colo/Documents\python/My projects - Data Science/Pokemon/pokemon_cleaned.csv')
combat = pd.read_csv('C:/Users/Colo/Documents\python/My projects - Data Science/Pokemon/combats.csv')

# Before I merge to datasets, I want to calculate the win percentage of each pokemon to use it in
# further part of project in building models

# calculating the win percentage of each pokemon
total_wins = combat['Winner'].value_counts()
# calculating total wins for each pokemon
numberofWins = combat.groupby('Winner').count()

# both methods produced the same results
countbyFirst = combat.groupby('Second_pokemon').count()
countbySecond = combat.groupby('First_pokemon').count()

print('Count by first winner shape:' + str(countbyFirst.shape))
print('Count by second winner shape:' + str(countbySecond.shape))
print('Total Wins shape: ' + str(total_wins.shape) )

# We can see that the number of dimensions is different in the total wins. 
# This can only mean there is one pokemon that was unable to win during it's fights. 
# Lets find the pokemon that did not win a single fight.

find_losing_pokemon = np.setdiff1d(countbyFirst.index.values, numberofWins.index.values)-1
losing_pokemon = pokemon.iloc[find_losing_pokemon[0],]
print(losing_pokemon)

# Shuckle, it appears that he has very strong defense but is very weak in all other catergories.
# Add Shuckle to the data so that the array lengths are the same. This will make merging the two datasets easier. 

# Now I am going to add new columns to data frame 'numberofWins'
numberofWins = numberofWins.sort_index()
numberofWins['Total Fights'] = countbyFirst.Winner + countbySecond.Winner
numberofWins['Win Percentage'] = numberofWins.First_pokemon/numberofWins['Total Fights']


# merge the original data set with winning data set
pokemon_merged = pd.merge(pokemon, numberofWins, left_on='Number', right_index = True, how='left')
data_for_ML = pd.merge(pokemon, numberofWins, right_index = True, left_on='Number')

# saving datasets

pokemon_merged.to_csv('C:/Users/Colo/Documents\python/My projects - Data Science/Pokemon/Pokemon_merged.csv')
data_for_ML.to_csv('C:/Users/Colo/Documents\python/My projects - Data Science/Pokemon/Pokemon_for_ML.csv')


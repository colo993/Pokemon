# In this part there is cleaning of data sets and visualization of 'Pokemon.csv' data set.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading data set
pokemon = pd.read_csv('C:/Users/Colo/Documents\python/My projects - Data Science/Pokemon/pokemon.csv')
combat = pd.read_csv('C:/Users/Colo/Documents\python/My projects - Data Science/Pokemon/combats.csv')

# checking information about data
pokemon.info()

# rename column with # sign to Number
pokemon = pokemon.rename(columns={'#':'Number'})

# There is some missing values in column 'Name' and 'Type 2'
pokemon['Name'].value_counts()

# We will try to find the name of missing pokemon. In dataframe we can find that missing name is between 61 and 63 index
print('This pokemon is before the missing pokemon:' + pokemon['Name'][61])
print(pokemon[pokemon["Name"].isnull()])
print('This pokemon is after the missing pokemon:' + pokemon['Name'][63])

# on site 'Pokedex' with pokemon list I found that pokemon between Mankey and Growlithe is Primeape! 
pokemon['Name'][62] = 'Primeape'

# Filling Nan values in column Type 2 with string 'None'
pokemon.fillna('None', inplace=True)
pokemon.info()

# Checking if there is any missing values in combat
combat.isnull().sum() # there is no missing values

# Saving cleaning dataframe as csv file
pokemon.to_csv('C:/Users/Colo/Documents\python/My projects - Data Science/Pokemon/pokemon_cleaned.csv')
##########################################################################################################################
# At this point, cleaning data is finished. Let's start with looking into the data

# Counting how many pokemons are from each group in Type 1 and Type 2
type1_pokemon = pd.DataFrame(pokemon['Type 1'].value_counts())
type2_pokemon = pd.DataFrame(pokemon['Type 2'].value_counts().iloc[1:20]) # iloc to delet None values

# Creating common data set for both types 
pokemon_types = type1_pokemon.join(type2_pokemon)

# plot amount of pokemons types 
plt.figure(figsize=(20, 8))
pokemon_types.plot(kind='bar')
plt.title('Amount of pokemons by Types')
plt.xlabel('Types')
plt.ylabel('Amount of pokemon')
plt.legend(loc='upper center')

# Plot legendary and nonlegendary pokemons by generation
plt.figure()
ax = sns.countplot(x='Generation', hue='Legendary', data=pokemon)
plt.title('Amount of pokemons in each generation')
plt.title('Amount of Pokemons by generation')

# Scatter plot def- atk by Type 1 for all pokemons
sns.relplot(x='Attack', y='Defense', hue='Type 1', data=pokemon)
plt.title('Dependance Defense of Attack for each pokemon by Type 1')

# Scatter plot def- atk by Type 1 for all pokemons diveded by Legendary status
sns.relplot(x='Attack', y='Defense', hue='Type 1',col='Legendary', data=pokemon)


# Creating dataset with type 1 and all statistics for each pokemon
pokemon_stats = pokemon[['Type 1','HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]

# plotting PairGrid stats of all pokemons
g = sns.PairGrid(pokemon_stats, hue='Type 1')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()



# Creating dataset with only water type 1 pokemon and all stats
water_stats = pokemon_stats.loc[pokemon_stats['Type 1'] == 'Water']

# plotting PairGrid stats of only water pokemons
g = sns.PairGrid(water_stats, hue='Type 1')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()

# Creating dataset with only water and fire type 1 pokemon and all stats
water_fire_stats = pokemon_stats.loc[pokemon_stats['Type 1'].isin(['Water', 'Fire'])]

# plotting PairGrid stats of only water pokemons
g = sns.PairGrid(water_fire_stats, hue='Type 1')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()


# box plot of all atributes 
# choosing only stats without Type_1 from dataframe pokemon_stats
x = pokemon_stats.iloc[:,1:]
# plot the values
g = sns.catplot(kind='box', data=x)
g.fig.suptitle('Statistics of all pokemons')


#box plot of HP atribute for each type of pokemon
g = sns.catplot(x = 'Type 1', y = 'HP', kind='box', data=pokemon_stats)
plt.xticks(rotation=90)
plt.title('HP value for each type of pokemon')

# then box plot of Attack atribute for each type of pokemon
g = sns.catplot(x = 'Type 1', y = 'Attack', kind='box', data=pokemon_stats)
plt.xticks(rotation=90)
plt.title('Attack value for each type of pokemon')

# then box plot of Defense atribute for each type of pokemon
g = sns.catplot(x = 'Type 1', y = 'Defense', kind='box', data=pokemon_stats)
plt.xticks(rotation=90)
plt.title('Defense value for each type of pokemon')

# Scatter plot of dependence between Attack na HP grouped by Generation
pokemon.plot(kind='scatter', x='Attack', y='HP',c='Generation', colormap='viridis')








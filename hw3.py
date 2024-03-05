"""
hw3.py
Name(s):
Date:
"""

import math
import random
import Organism as Org
import matplotlib.pyplot as plt
import numpy as np

"""
crossover operation for genetic algorithm

INPUTS:
parent1: first parent
parent2 : second parent


OUTPUTS:
genome1: genome for first child
genome2: second for second child

This funciton creates two genomes from parents and returns them
"""
def crossover(parent1, parent2):
    # create a random index k to split into parents' genomes
    k = round(random.random() * len(parent1.bits))

    # generate two children's genomes combining parents' genomes
    genome1 = np.hstack((parent1.bits[0:k], parent2.bits[k:]))
    genome2 = np.hstack((parent2.bits[0:k], parent1.bits[k:]))

    return (genome1, genome2)

"""
mutation operation for genetic algorithm
"""
def mutation(genome, mutRate):
    # iterate through every bit in genome
    for bit in genome:
        # flip it with probability of mutRate
        if (random.random() < mutRate):
            bit = 1 - bit

    return genome

"""
selection operation for choosing a parent for mating from the population
"""
def selection(pop):
    # select a random number
    rand = random.random()

    # find the first organism with accumulated fitness larger than random number
    for org in pop:
        if org.accFit > rand:
            return org
    
    # if did not find such an organis, return the last organism in population
    return pop[-1]

"""
calcFit will calculate the fitness of an organism
"""
def calcFit(org, xVals, yVals):
    # # I believe this would be a way to make the algorithm more efficient
    # #if fitness for this organism was already calculated, just return it
    # if org.fitness != 0:
    #     return org.fitness
    
    # Create a variable to store the running sum error.
    error = 0

    # Loop over each x value.
    for ind in range(len(xVals)):
        # Create a variable to store the running sum of the y value.
        y = 0
        
        # Compute the corresponding y value of the fit by looping
        # over the coefficients of the polynomial.
        for n in range(len(org.floats)):
            # Add the term c_n*x^n, where c_n are the coefficients.
            try:
                y += org.floats[n] * (xVals[ind])**n
            except OverflowError:
                y += math.inf

        # Compute the squared error of the y values, and add to the running
        # sum of the error.
        try:
            error += (y - yVals[ind])**2
        except OverflowError:
            error += math.inf

    # Now compute the sqrt(error), average it over the data points,
    # and return the reciprocal as the fitness.
    if error == 0:
        return math.inf
    else:
        fitness = len(xVals)/math.sqrt(error)
        if not math.isnan(fitness):
            return fitness
        else:
            return 0

"""
accPop will calculate the fitness and accFit of the population
"""
def accPop(pop, xVals, yVals):
    # create a totalfit variable used later to normalize individual fitness values
    totalfit = 0.

    # calculate fitness for each organism in the population, increment totalfit
    for org in pop:
        thisfitness = calcFit(org, xVals, yVals)
        org.fitness = thisfitness
        totalfit += thisfitness

    # sort the
    # population by descending fitness values
    pop.sort(reverse=True)

    # create a variable to keep track of accumulated fitness so far
    accumulated = 0.

    #for each organism, calculate normalized fitness and accumulated fitness
    for org in pop:
        org.normFit = org.fitness / totalfit
        accumulated += org.normFit
        org.accFit = accumulated

    return pop

"""
initPop will initialize a population of a given size and number of coefficients
"""
def initPop(size, numCoeffs):
    # Get size-4 random organisms in a list.
    pop = [Org.Organism(numCoeffs) for x in range(size-4)]

    # Create the all 0s and all 1s organisms and append them to the pop.
    pop.append(Org.Organism(numCoeffs, [0]*(64*numCoeffs)))
    pop.append(Org.Organism(numCoeffs, [1]*(64*numCoeffs)))

    # Create an organism corresponding to having every coefficient as 1.
    bit1 = [0]*2 + [1]*10 + [0]*52
    org = []
    for c in range(numCoeffs):
        org = org + bit1
    pop.append(Org.Organism(numCoeffs, org))

    # Create an organism corresponding to having every coefficient as -1.
    bit1 = [1,0] + [1]*10 + [0]*52
    org = []
    for c in range(numCoeffs):
        org = org + bit1
    pop.append(Org.Organism(numCoeffs, org))

    # Return the population.
    return pop

"""
nextGeneration will create the next generation
"""
def nextGeneration(pop, numCoeffs, mutRate, eliteNum):
    # create a new population variable
    newPop = []

    # perform matings to create children and form the new pop
    for i in range((len(pop)-eliteNum)//2):

        # select two parents
        parent1 = selection(pop)
        parent2 = selection(pop)

        # generate two new genomes from parents
        (genome1, genome2) = crossover(parent1, parent2)

        # mutate the two genomes
        genome1 = mutation(genome1, mutRate)
        genome2 = mutation(genome2, mutRate)

        # create two children using two new genomes
        child1 = Org.Organism(numCoeffs, genome1)
        child2 = Org.Organism(numCoeffs, genome2)

        # add new children to the new population
        newPop.append(child1)
        newPop.append(child2)

    # copy the elite into the new population
    for i in range(eliteNum):
        newPop.append(pop[i])

    return newPop

"""
GA will perform the genetic algorithm for k+1 generations (counting
the initial generation).

INPUTS
k:         the number of generations
size:      the size of the population
numCoeffs: the number of coefficients in our polynomials
mutRate:   the mutation rate
xVals:     the x values for the fitting
yVals:     the y values for the fitting
eliteNum:  the number of elite individuals to keep per generation
bestN:     the number of best individuals to track over time

OUTPUTS
best: the bestN number of best organisms seen over the course of the GA
fit:  the highest observed fitness value for each iteration
"""
def GA(k, size, numCoeffs, mutRate, xVals, yVals, eliteNum, bestN):
    # create initial population and sort it by fitness
    pop = initPop(size, numCoeffs)
    pop = accPop(pop, xVals, yVals)

    # initialize and populate the list of best organisms
    best = []
    for ind in range(bestN):
        best.append(pop[ind])

    #initialize the list of highest fitness for each generation
    fit = []
    #record the highest fitness for the initial generation
    fit.append(pop[0].fitness)

    # keep creating generations and repeating the steps
    for generation in range(k):
        print(generation) # debugging
        # create new population and sort it by fitness
        newPop = nextGeneration(pop, numCoeffs, mutRate, eliteNum)
        newPop = accPop(newPop, xVals, yVals)


        # update the list of best organisms
        #for the first N organisms 
        for ind in range(bestN):
            #record the organism in new population
            thisOrg = newPop[ind]

            # assume it is not recorded in the list of best organisms
            recorded = False

            #check if it is, comparing it to all orgs in the list of best
            for bestOrg in best:
                if (thisOrg.isClone(bestOrg)):
                    recorded = True

            # if not recorded, add it to the list of best
            if (not recorded):
                best.append(thisOrg)

        #sort the list of best organisms
        best.sort(reverse=True)
            
        # truncate the list of best to only keep N organisms
        best = best[0:bestN] 


        # record the highest fitness for this generation
        fit.append(newPop[0].fitness)

        # update current population
        pop = newPop
    
    return (best,fit)

"""
runScenario will run a given scenario, plot the highest fitness value for each
generation, and return a list of the bestN number of top individuals observed.

INPUTS
scenario: a string to use for naming output files.
--- the remaining inputs are those for the call to GA ---

OUTPUTS
best: the bestN number of best organisms seen over the course of the GA
--- Plots are saved as: 'fit' + scenario + '.png' ---
"""
def runScenario(scenario, k, size, numCoeffs, mutRate, \
                xVals, yVals, eliteNum, bestN):

    # Perform the GA.
    (best,fit) = GA(k, size, numCoeffs, mutRate, xVals, yVals, eliteNum, bestN)

    # Plot the fitness per generation.
    gens = range(k+1)
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(gens, fit)
    plt.title('Best Fitness per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.savefig('fit'+scenario+'.png', bbox_inches='tight')
    plt.close('all')

    # Return the best organisms.
    return best

"""
main function
"""
if __name__ == '__main__':

    # Flags to suppress any given scenario. Simply set to False and that
    # scenario will be skipped. Set to True to enable a scenario.
    scenA = True
    scenB = True
    scenC = True
    scenD = True

    if not (scenA or scenB or scenC or scenD):
        print("All scenarios disabled. Set a flag to True to run a scenario.")
    
################################################################################
    ### Scenario A: Fitting to a constant function, y = 1. ###
################################################################################

    if scenA:
        # Create the x values ranging from 0 to 10 with a step of 0.1.
        xVals = [0.1*n for n in range(101)]

        # Create the y values for y = 1 corresponding to the x values.
        yVals = [1. for n in range(len(xVals))]

        # Set the other parameters for the GA.
        sc = 'A'      # Set the scenario title.
        k = 100       # 100 generations for our GA.
        size = 1000   # Population of 1000.
        numCoeffs = 4 # Cubic polynomial.
        mutRate = 0.1 # 10% mutation rate.
        eliteNum = 50 # Keep the top 5% of the population.
        bestN = 5     # track the top 5 seen so far.

        # Run the scenario.
        best = runScenario(sc, k, size, numCoeffs, mutRate, xVals, yVals, \
                           eliteNum, bestN)

        # Print the best individuals.
        print()
        print('Best individuals of Scenario '+ sc +':')
        for org in best:
            print(org)
            print()
            print()
            print()

################################################################################
    ### Scenario B: Fitting to a constant function, y = 5. ###
################################################################################
    
    if scenB:
        # Create the x values ranging from 0 to 10 with a step of 0.1.
        xVals = [0.1*n for n in range(101)]

        # Create the y values for y = 1 corresponding to the x values.
        yVals = [5. for n in range(len(xVals))]

        # Set the other parameters for the GA.
        sc = 'B'      # Set the scenario title.
        k = 250       # 250 generations for our GA.
        size = 1000   # Population of 1000.
        numCoeffs = 4 # Cubic polynomial.
        mutRate = 0.1 # 10% mutation rate.
        eliteNum = 50 # Keep the top 5% of the population.
        bestN = 5     # track the top 5 seen so far.

        # Run the scenario.
        best = runScenario(sc, k, size, numCoeffs, mutRate, xVals, yVals, \
                           eliteNum, bestN)

        # Print the best individuals.
        print()
        print('Best individuals of Scenario '+ sc +':')
        for org in best:
            print(org)
            print()
            print()
            print()

################################################################################
    ### Scenario C: Fitting to a quadratic function, y = x^2 - 1. ###
################################################################################
    
    if scenC:
        # Create the x values ranging from 0 to 10 with a step of 0.1.
        xVals = [0.1*n for n in range(101)]

        # Create the y values for y = x^2 - 1 corresponding to the x values.
        yVals = [x**2-1. for x in xVals]

        # Set the other parameters for the GA.
        sc = 'C'      # Set the scenario title.
        k = 250       # 250 generations for our GA.
        size = 1000   # Population of 1000.
        numCoeffs = 4 # Cubic polynomial.
        mutRate = 0.1 # 10% mutation rate.
        eliteNum = 50 # Keep the top 5% of the population.
        bestN = 5     # track the top 5 seen so far.

        # Run the scenario.
        best = runScenario(sc, k, size, numCoeffs, mutRate, xVals, yVals, \
                           eliteNum, bestN)

        # Print the best individuals.
        print()
        print('Best individuals of Scenario '+ sc +':')
        for org in best:
            print(org)
            print()
            print()
            print()

################################################################################
    ### Scenario D: Fitting to a quadratic function, y = cos(x). ###
################################################################################
    
    if scenD:
        # Create the x values ranging from -5 to 5 with a step of 0.1.
        xVals = [0.1*n-5 for n in range(101)]

        # Create the y values for y = cos(x) corresponding to the x values.
        yVals = [math.cos(x) for x in xVals]

        # Set the other parameters for the GA.
        sc = 'D'      # Set the scenario title.
        k = 250       # 250 generations for our GA.
        size = 1000   # Population of 1000.
        numCoeffs = 5 # Quartic polynomial with 4 zeros!
        mutRate = 0.1 # 10% mutation rate.
        eliteNum = 50 # Keep the top 5% of the population.
        bestN = 5     # track the top 5 seen so far.

        # Run the scenario.
        best = runScenario(sc, k, size, numCoeffs, mutRate, xVals, yVals, \
                           eliteNum, bestN)

        # Print the best individuals.
        print()
        print('Best individuals of Scenario '+ sc +':')
        for org in best:
            print(org)
            print()
            print()
            print()

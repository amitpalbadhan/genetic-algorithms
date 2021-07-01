#include "stringmatching.hpp"


void StringMatching::setTargetString(std::string inputString) {
    targetString = inputString;
}


int StringMatching::getTargetSize() {
    return targetSize;
}


StringMatching::StringMatching (std::string inputString) {
    setTargetString(inputString);
    targetSize = inputString.size();

    // seed mt19937
    rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());

    // init random number generator (max 255 for characters)
    distrib = std::uniform_int_distribution<int>(0, 255);
}


void StringMatching::initializePopulation(int initialSize) {
    populationSize = initialSize;
    population.resize(populationSize);

    for (int i = 0; i < populationSize; ++i) {
        std::string currentOffspring = "";
        
        for (int j = 0; j < targetSize; ++j) {
            currentOffspring += (char)distrib(rng);
        }

        population[i] = currentOffspring;
    }
}


int StringMatching::getFitness(std::string offspring) {
    int fitness = 0;

    for (int i = 0; i < offspring.size(); ++i) {
        if (offspring[i] == targetString[i]) {
            ++fitness;
        }
    }
    
    return fitness;
}


std::string StringMatching::getFittest(std::vector<std::string> currentPopulation) {
    std::string fittestOffspring;

    int maxFitnessValue = -1;
    int fittestIdx = 0;

    for (int i = 0; i < currentPopulation.size(); ++i) {
        int currentFitness = getFitness(currentPopulation[i]);
        if (currentFitness > maxFitnessValue) {
            maxFitnessValue = currentFitness;
            fittestIdx = i;
        }
    }

    return currentPopulation[fittestIdx];
}


std::string StringMatching::getFittestChromosome() {
    return getFittest(population);
}


int StringMatching::getFitnessOfFittest() {
    return getFitness(getFittest(population));
}


std::string StringMatching::uniformCrossover(std::string parent1, std::string parent2) {
    // randomly select a gene from either parent

    std::string offspring = "";

    for (int i = 0; i < targetSize; ++i) {
        if (distrib(rng) < 0.5 * 255) {
            // gene from parent1
            offspring += parent1[i];
        } else {
            // gene from parent2
            offspring += parent2[i];
        }
    }

    return offspring;
}


std::string StringMatching::mutate(std::string offspring) {
    // randomly mutate an offspring
    std::string mutatedOffspring = offspring;

    
    for (int i = 0; i < targetSize; ++i) {
        if (distrib(rng) < mutationRate * 255) {
            mutatedOffspring[i] = (char)distrib(rng);
        }
    }
    
    return mutatedOffspring;
}


std::string StringMatching::tournamentSelection(std::vector<std::string> currentPopulation) {
    std::vector<std::string> tournament(tournamentSize);
    std::vector<bool> inTournament(tournamentSize, false);

    std::uniform_int_distribution<> select(0, currentPopulation.size() - 1);

    for (int i = 0; i < tournamentSize; ++i) {
        int randomOffspring;
        do {
            randomOffspring = select(rng);
        } while (inTournament[randomOffspring]);

        inTournament[randomOffspring] = true;
        tournament.push_back(currentPopulation[randomOffspring]);
    }

    return getFittest(tournament);
}


void StringMatching::evolvePopulation() {
    std::vector<std::string> newPopulation(populationSize);

    if (elitism) {
        newPopulation[0] = getFittest(population);
    }

    int elitismOffset;
    if (elitism) elitismOffset = 1;
    else elitismOffset = 0;

    for (int i = elitismOffset; i < populationSize; ++i) {
        std::string parent1 = tournamentSelection(population);
        std::string parent2 = tournamentSelection(population);
        std::string offspring = uniformCrossover(parent1, parent2);
        newPopulation[i] = offspring;
    }

    // mutate population
    for (int i = elitismOffset; i < population.size(); ++i) {
        newPopulation[i] = mutate(newPopulation[i]);
    }

    population = newPopulation;
}

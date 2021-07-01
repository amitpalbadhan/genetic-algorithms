#pragma once

#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <vector>

class StringMatching {
private:
    // constants
    const double mutationRate = 0.05;
    const int tournamentSize = 5;
    const bool elitism = true;

    int targetSize;
    int populationSize;
    int maxFitness;
    std::string targetString;
    std::vector<std::string> population;
    std::mt19937 rng;
    std::uniform_int_distribution<> distrib;
public:
    void setTargetString(std::string inputString);


    int getTargetSize();


    StringMatching (std::string inputString);


    void initializePopulation(int initialSize);


    int getFitness(std::string offspring);


    std::string getFittest(std::vector<std::string> currentPopulation);


    std::string getFittestGene();


    int getFitnessOfFittest();


    std::string uniformCrossover(std::string parent1, std::string parent2);


    std::string mutate(std::string offspring);


    std::string tournamentSelection(std::vector<std::string> currentPopulation);


    void evolvePopulation();
};
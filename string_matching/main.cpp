#include <iostream>

#include "stringmatching.hpp"


int main() {
    std::cout << "Target string: ";
    std::string targetString;
    getline(std::cin, targetString);
    StringMatching stringMatching(targetString);

    stringMatching.initializePopulation(50);

    int generationCount = 0;
    while (stringMatching.getFitnessOfFittest() < stringMatching.getTargetSize()) {
        ++generationCount;
        
        // prints the fittest chromosome of each generation
        // std::cout << stringMatching.getFittestChromosome() << "\n";
        
        // prints the highest fittness score of each generation
        std::cout << "Generation: " << generationCount << ", Fittest: " << stringMatching.getFitnessOfFittest() << "\n";
        
        stringMatching.evolvePopulation();
    }

    std::cout << "Solution Found\n";
    std::cout << "Generation: " << generationCount << "\n\n";

    std::cout << "Target: " << targetString << "\n";
    std::cout << "Solution Found: " << stringMatching.getFittestGene() << "\n";

    return 0;
}

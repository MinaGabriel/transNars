#ifndef GLOBALS_H
#define GLOBALS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <chrono>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>

#define INT int64_t
#define REAL float
#define MAX_THREADS 5

struct Triple
{
    INT h, r, t;
    static bool cmp_head(const Triple &a, const Triple &b)
    {
        return (a.h < b.h) || (a.h == b.h && a.r < b.r) || (a.h == b.h && a.r == b.r && a.t < b.t);
    }
    static bool cmp_tail(const Triple &a, const Triple &b)
    {
        return (a.t < b.t) || (a.t == b.t && a.r < b.r) || (a.t == b.t && a.r == b.r && a.h < b.h);
    }
};

Triple *trainList;

Triple *trainHead; //sorted by Head, Relation and Tail order
Triple *trainTail; //Sorted by Tail, Relation and Head order

std::string inPath;
INT relationTotal;
INT entityTotal;
INT trainTotal;
INT testTotal;
INT validTotal;
INT tripleTotal;

REAL *headCorruptProb;
INT *headStartIndices = nullptr, *headEndIndices = nullptr;
INT *tailStartIndices = nullptr, *tailEndIndices = nullptr;

Triple *testList;
Triple *validList;
Triple *tripleList;

INT *testRelationStartIndex;
INT *testRelationEndIndex;
INT *validRelationStartIndex;
INT *validRelationEndIndex;

std::vector<REAL> left_mean, right_mean;
unsigned long long *next_random;

// Function to convert a number to a human-readable string with suffixes
std::string humanReadableNumber(long long num) {
    const char* suffixes[] = {"", "K", "M", "B", "T"};
    size_t suffixIndex = 0;
    double reducedNum = num;

    // Reduce the number and increment the suffix index
    while (reducedNum >= 1000 && suffixIndex < sizeof(suffixes)/sizeof(suffixes[0]) - 1) {
        reducedNum /= 1000.0;
        suffixIndex++;
    }

    // Create a string stream to format the number
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << reducedNum << suffixes[suffixIndex];
    return ss.str();
}

class Random
{
public:
    static void setSeed(unsigned int seed);
    static uint64_t randd(int threadIndex);
    static int randMax(int threadIndex, int x);

private:
    static uint64_t next_random[MAX_THREADS];
};

uint64_t Random::next_random[MAX_THREADS];

void Random::setSeed(unsigned int seed)
{
    unsigned int local_seed = seed;
    std::generate(next_random, next_random + MAX_THREADS, [&local_seed]()
                  { return rand_r(&local_seed); });
}

uint64_t Random::randd(int threadIndex)
{
    next_random[threadIndex] = next_random[threadIndex] * (uint64_t)(25214903917) + 11;
    return next_random[threadIndex];
}

int Random::randMax(int threadIndex, int x)
{
    int res = static_cast<int>(randd(threadIndex) % x);
    while (res < 0)
        res += x;
    return res;
}

static INT readFirstLine(const std::string &filePath)
{
    int total = 0;
    std::ifstream fin(filePath);
    if (!fin.is_open())
    {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return -1;
    }

    if (!(fin >> total))
    {
        std::cerr << "Error reading the first line as a number from file: " << filePath << std::endl;
        return -1;
    }

    fin.close();
    return total;
}

static void writeTriplesToFile(const std::string &filePath, Triple *triples, INT count) 
{
    std::ofstream outFile(filePath);
    if (!outFile)
    {
        std::cerr << "Error opening file for writing: " << filePath << std::endl;
        return;
    }
    for (INT i = 0; i < count; ++i)
    {
        outFile << triples[i].h << "\t" << triples[i].r << "\t" << triples[i].t << "\n";
    }
    outFile.close();
}



#endif // GLOBALS_H
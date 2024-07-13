
#include "Globals.h"

void importValidFile(INT *batch_h, INT *batch_t, INT *batch_r)
{
    auto start = std::chrono::high_resolution_clock::now();


    validTotal = readFirstLine(inPath + "valid2id.txt");
 
    validList = (Triple *)calloc(validTotal, sizeof(Triple)); 

    // Read valid data
    {
        std::ifstream fin(inPath + "valid2id.txt");
        std::string firstLine;
        std::getline(fin, firstLine); // Skip the first line
        for (int i = 0; i < validTotal; ++i)
        {
            fin >> validList[i].h >> validList[i].t >> validList[i].r;
            batch_h[i] = validList[i].h;
            batch_t[i] = validList[i].t;
            batch_r[i] = validList[i].r;
        }
        fin.close();
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
}
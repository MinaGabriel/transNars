#include "Globals.h"
#include "Train.h"
#include "Test.h"
#include "Valid.h"
extern "C" void testDataLoader(char *path, INT *batch_h, INT *batch_t, INT *batch_r){
    inPath = path;
    relationTotal = readFirstLine(inPath + "relation2id.txt");
    entityTotal = readFirstLine(inPath + "entity2id.txt");
    importTestFiles(batch_h, batch_t, batch_r);
    std::cout << "C++ Training Data Done!" << std::endl; 

}


extern "C" void validDataLoader(char *path, INT *batch_h, INT *batch_t, INT *batch_r){
    inPath = path;
    importValidFile(batch_h, batch_t, batch_r);
    std::cout << "C++ Validation Data Done!" << std::endl; 

}


extern "C" void trainDataLoader(char *path, INT *batch_h, INT *batch_t, INT *batch_r, REAL *batch_y, INT neg_ratio, INT max_threads)
{
    auto start = std::chrono::high_resolution_clock::now();

    inPath = path;
    relationTotal = readFirstLine(inPath + "relation2id.txt");
    entityTotal = readFirstLine(inPath + "entity2id.txt");
    importTrainFiles();

    std::cout << "relationTotal: " << relationTotal << std::endl;
    std::cout << "entityTotal: " << entityTotal << std::endl;
    std::cout << "trainTotal: " << trainTotal << std::endl;
    std::cout << "neg_ratio: " << neg_ratio << std::endl;

    // std::cout << "headCorruptProb" << ": [";
    // for (int i = 0; i < relationTotal; i++)
    // {
    //     std::cout << headCorruptProb[i] << " ";
    // }
    // std::cout << "]" << std::endl;

    INT workThreads = max_threads;
    INT total = trainTotal * (1 + neg_ratio);
    std::cout << "Total with neg samples: " << total << std::endl;
    std::cout << "WorkThreads: " << workThreads << std::endl;
    while (total % workThreads > 0)
    {   
        workThreads--;
        std::cout << " WorkThreads - 1 = " << workThreads << std::endl;
        

    }

    if (!batch_h || !batch_t || !batch_r || !batch_y)
    {
        std::cerr << "Error: Memory allocation failed." << std::endl;
    }

    pthread_t *pt = (pthread_t *)malloc(workThreads * sizeof(pthread_t));
    Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));

    if (!pt || !para)
    {
        std::cerr << "Error: Memory allocation failed." << std::endl;
    }

    next_random = (unsigned long long *)malloc(workThreads * sizeof(unsigned long long));
    if (!next_random)
    {
        std::cerr << "Error: Memory allocation failed." << std::endl;
    }

    INT chunkSize = total / workThreads;
    for (INT threads = 0; threads < workThreads; threads++)
    {
        para[threads].id = threads;
        para[threads].batch_h = batch_h;
        para[threads].batch_t = batch_t;
        para[threads].batch_r = batch_r;
        para[threads].batch_y = batch_y;
        para[threads].negRate = neg_ratio;
        para[threads].startIdx = threads * chunkSize;
        para[threads].endIdx = (threads == workThreads - 1) ? total : (threads + 1) * chunkSize;
        para[threads].trainList = trainList;
        para[threads].trainTotal = trainTotal;
        para[threads].headCorruptProb = headCorruptProb;

        pthread_create(&pt[threads], NULL, getBatch, (void *)&para[threads]);
    }

    for (INT threads = 0; threads < workThreads; threads++)
    {
        int rc = pthread_join(pt[threads], NULL);
        if (rc != 0)
        {
            fprintf(stderr, "Error: Unable to join thread %ld, return code %d\n", threads, rc);
        }
    }

    printf("Total Size = %ld\n", total);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    // TODO: Add remove for batches (batch_h, batch_t etc ...)
    free(pt);
    free(para);
    free(next_random);
}

int main()
{
    // database path
    inPath = "../../datasets/FB15K237/";
    relationTotal = readFirstLine(inPath + "relation2id.txt");
    entityTotal = readFirstLine(inPath + "entity2id.txt");
    importTrainFiles();

    for (int i = 0; i < trainTotal; i++)
    {
        std::cout << trainList[i].h << " " << trainList[i].h
                  << " " << trainList[i].t << std::endl;
        break;
    }
}
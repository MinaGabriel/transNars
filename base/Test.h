#include "Globals.h"


INT lastHead = 0;
INT lastTail = 0;
REAL l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
REAL l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;
REAL hit1 = 0, hit3 = 0, hit10 = 0, mr = 0, mrr = 0;

bool _find(INT h, INT t, INT r)
{
    INT lef = 0;
    INT rig = tripleTotal - 1;
    while (lef + 1 < rig)
    {
        INT mid = (lef + rig) >> 1;
        if ((tripleList[mid].h < h) || (tripleList[mid].h == h && tripleList[mid].r < r) || (tripleList[mid].h == h && tripleList[mid].r == r && tripleList[mid].t < t))
            lef = mid;
        else
            rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t)
        return true;
    if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t)
        return true;
    return false;
}

extern "C" void testHead(REAL *con, INT lastHead)
{
    INT h = testList[lastHead].h;
    INT t = testList[lastHead].t;
    INT r = testList[lastHead].r;

    REAL minimal = con[h]; 
    INT l_s = 0;
    INT l_filter_s = 0;

    for (INT j = 0; j < entityTotal; j++)
    {
        if (j != h)
        {
            REAL value = con[j];
            if (value < minimal)
            {
                l_s += 1;
                if (not _find(j, t, r))
                    l_filter_s += 1;
            }
        }
    }

    if (l_filter_s < 10) l_filter_tot += 1;
    if (l_s < 10) l_tot += 1;
    if (l_filter_s < 3) l3_filter_tot += 1;
    if (l_s < 3) l3_tot += 1;
    if (l_filter_s < 1) l1_filter_tot += 1;
    if (l_s < 1) l1_tot += 1;

    l_filter_rank += (l_filter_s+1);
    l_rank += (1 + l_s);
    l_filter_reci_rank += 1.0/(l_filter_s+1);
    l_reci_rank += 1.0/(l_s+1);
}



extern "C" void testTail(REAL *con, INT lastTail) {
    INT h = testList[lastTail].h;
    INT t = testList[lastTail].t;
    INT r = testList[lastTail].r;
    
    REAL minimal = con[t];
    INT r_s = 0;
    INT r_filter_s = 0;
    for (INT j = 0; j < entityTotal; j++) {
        if (j != t) {
            REAL value = con[j];
            if (value < minimal) {
                r_s += 1;
                if (not _find(h, j, r))
                    r_filter_s += 1;
            }
        }
        
    }

    if (r_filter_s < 10) r_filter_tot += 1;
    if (r_s < 10) r_tot += 1;
    if (r_filter_s < 3) r3_filter_tot += 1;
    if (r_s < 3) r3_tot += 1;
    if (r_filter_s < 1) r1_filter_tot += 1;
    if (r_s < 1) r1_tot += 1;

    r_filter_rank += (1+r_filter_s);
    r_rank += (1+r_s);
    r_filter_reci_rank += 1.0/(1+r_filter_s);
    r_reci_rank += 1.0/(1+r_s);
}

extern "C"
void test_link_prediction() {
    l_rank /= testTotal;
    r_rank /= testTotal;
    l_reci_rank /= testTotal;
    r_reci_rank /= testTotal;
 
    l_tot /= testTotal;
    l3_tot /= testTotal;
    l1_tot /= testTotal;
 
    r_tot /= testTotal;
    r3_tot /= testTotal;
    r1_tot /= testTotal;

    // with filter
    l_filter_rank /= testTotal;
    r_filter_rank /= testTotal;
    l_filter_reci_rank /= testTotal;
    r_filter_reci_rank /= testTotal;
 
    l_filter_tot /= testTotal;
    l3_filter_tot /= testTotal;
    l1_filter_tot /= testTotal;
 
    r_filter_tot /= testTotal;
    r3_filter_tot /= testTotal;
    r1_filter_tot /= testTotal;

    printf("no type constraint results:\n");
    
    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
    printf("H(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_reci_rank, l_rank, l_tot, l3_tot, l1_tot);
    printf("T(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_reci_rank, r_rank, r_tot, r3_tot, r1_tot);
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
            (l_reci_rank+r_reci_rank)/2, (l_rank+r_rank)/2, (l_tot+r_tot)/2, (l3_tot+r3_tot)/2, (l1_tot+r1_tot)/2);
    printf("\n");
    printf("H(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_filter_reci_rank, l_filter_rank, l_filter_tot, l3_filter_tot, l1_filter_tot);
    printf("T(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_filter_reci_rank, r_filter_rank, r_filter_tot, r3_filter_tot, r1_filter_tot);
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
            (l_filter_reci_rank+r_filter_reci_rank)/2, (l_filter_rank+r_filter_rank)/2, (l_filter_tot+r_filter_tot)/2, (l3_filter_tot+r3_filter_tot)/2, (l1_filter_tot+r1_filter_tot)/2);

    mrr = (l_filter_reci_rank+r_filter_reci_rank) / 2;
    mr = (l_filter_rank+r_filter_rank) / 2;
    hit10 = (l_filter_tot+r_filter_tot) / 2;
    hit3 = (l3_filter_tot+r3_filter_tot) / 2;
    hit1 = (l1_filter_tot+r1_filter_tot) / 2;


}


void importTestFiles(INT *batch_h, INT *batch_t, INT *batch_r)
{
    auto start = std::chrono::high_resolution_clock::now();

    // Initialize global variables
    relationTotal = readFirstLine(inPath + "relation2id.txt");
    entityTotal = readFirstLine(inPath + "entity2id.txt");

    testTotal = readFirstLine(inPath + "test2id.txt");
    trainTotal = readFirstLine(inPath + "train2id.txt");
    validTotal = readFirstLine(inPath + "valid2id.txt");

    tripleTotal = testTotal + trainTotal + validTotal;

    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    validList = (Triple *)calloc(validTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));

    // Read test data
    {
        std::ifstream fin(inPath + "test2id.txt");
        std::string firstLine;
        std::getline(fin, firstLine); // Skip the first line
        for (int i = 0; i < testTotal; ++i)
        {
            fin >> testList[i].h >> testList[i].t >> testList[i].r;
            tripleList[i] = testList[i];
            batch_h[i] = tripleList[i].h;
            batch_t[i] = tripleList[i].t;
            batch_r[i] = tripleList[i].r;
        }
        fin.close();
    }

    // Read train data
    {
        std::ifstream fin(inPath + "train2id.txt");
        std::string firstLine;
        std::getline(fin, firstLine); // Skip the first line
        for (int i = 0; i < trainTotal; ++i)
        {
            fin >> tripleList[i + testTotal].h >> tripleList[i + testTotal].t >> tripleList[i + testTotal].r;
        }
        fin.close();
    }

    // Read valid data
    {
        std::ifstream fin(inPath + "valid2id.txt");
        std::string firstLine;
        std::getline(fin, firstLine); // Skip the first line
        for (int i = 0; i < validTotal; ++i)
        {
            fin >> validList[i].h >> validList[i].t >> validList[i].r;
            tripleList[i + testTotal + trainTotal] = validList[i];
        }
        fin.close();
    }

    // Sort triples
    std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_head);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
}
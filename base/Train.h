

INT corrupt_tail(INT id, INT h, INT r)
{
    INT i_start, i_end, i_mid;
    INT hr_start, hr_end;

    i_start = headStartIndices[h] - 1;
    i_end = headEndIndices[h];
    while (i_start + 1 < i_end)
    {
        i_mid = (i_start + i_end) >> 1;
        if (trainHead[i_mid].r >= r)
            i_end = i_mid;
        else
            i_start = i_mid;
    }
    hr_start = i_end;

    i_start = headStartIndices[h];
    i_end = headEndIndices[h] + 1;
    while (i_start + 1 < i_end)
    {
        i_mid = (i_start + i_end) >> 1;
        if (trainHead[i_mid].r <= r)
            i_start = i_mid;
        else
            i_end = i_mid;
    }
    hr_end = i_start;

    INT tmp = Random::randMax(id, entityTotal - (hr_end - hr_start + 1));
    if (tmp < trainHead[hr_start].t)
        return tmp;
    if (tmp > trainHead[hr_end].t - hr_end + hr_start - 1)
        return tmp + hr_end - hr_start + 1;
    i_start = hr_start, i_end = hr_end + 1;
    while (i_start + 1 < i_end)
    {
        i_mid = (i_start + i_end) >> 1;
        if (trainHead[i_mid].t - i_mid + hr_start - 1 < tmp)
            i_start = i_mid;
        else
            i_end = i_mid;
    }
    return tmp + i_start - hr_start + 1;
}

INT corrupt_head(INT id, INT t, INT r)
{
    INT i_start, i_end, i_mid;
    INT tr_start, tr_end;

    i_start = tailStartIndices[t] - 1;
    i_end = tailEndIndices[t];
    while (i_start + 1 < i_end)
    {
        i_mid = (i_start + i_end) >> 1;
        if (trainTail[i_mid].r >= r)
            i_end = i_mid;
        else
            i_start = i_mid;
    }
    tr_start = i_end;

    i_start = tailStartIndices[t];
    i_end = tailEndIndices[t] + 1;
    while (i_start + 1 < i_end)
    {
        i_mid = (i_start + i_end) >> 1;
        if (trainTail[i_mid].r <= r)
            i_start = i_mid;
        else
            i_end = i_mid;
    }
    tr_end = i_start;

    INT tmp = Random::randMax(id, entityTotal - (tr_end - tr_start + 1));
    if (tmp < trainTail[tr_start].h)
        return tmp;
    if (tmp > trainTail[tr_end].h - tr_end + tr_start - 1)
        return tmp + tr_end - tr_start + 1;
    i_start = tr_start, i_end = tr_end + 1;
    while (i_start + 1 < i_end)
    {
        i_mid = (i_start + i_end) >> 1;
        if (trainTail[i_mid].h - i_mid + tr_start - 1 < tmp)
            i_start = i_mid;
        else
            i_end = i_mid;
    }
    return tmp + i_start - tr_start + 1;
}

typedef struct
{
    INT id;
    INT *batch_h;
    INT *batch_t;
    INT *batch_r;
    REAL *batch_y;
    INT negRate;
    INT startIdx;
    INT endIdx;
    REAL *headCorruptProb;
    const Triple *trainList;
    INT trainTotal;
} Parameter;

void *getBatch(void *con)
{
    Parameter *para = (Parameter *)con;

    INT id = para->id;
    INT *batch_h = para->batch_h;
    INT *batch_t = para->batch_t;
    INT *batch_r = para->batch_r;
    REAL *batch_y = para->batch_y;
    INT negRate = para->negRate;
    INT startIdx = para->startIdx;
    INT endIdx = para->endIdx;
    REAL *headCorruptProb = para->headCorruptProb;
    const Triple *trainList = para->trainList;
    INT trainTotal = para->trainTotal;

    std::default_random_engine generator;
    std::uniform_int_distribution<INT> distribution(0, entityTotal - 1);

    for (INT batch = startIdx; batch < endIdx; batch += (1 + negRate))
    {
        if (batch >= trainTotal * (1 + negRate))
        {
            std::cerr << "Error: batch index " << batch << " out of bounds" << std::endl;
            continue;
        }

        INT dataIdx = (batch / (1 + negRate));
        dataIdx = dataIdx % trainTotal;

        if (dataIdx >= trainTotal)
        {
            std::cerr << "Error: dataIdx " << dataIdx << " out of bounds" << std::endl;
            continue;
        }

        batch_h[batch] = trainList[dataIdx].h;
        batch_t[batch] = trainList[dataIdx].t;
        batch_r[batch] = trainList[dataIdx].r;
        batch_y[batch] = 1;

        for (INT i = 1; i <= negRate; i++)
        {
            if ((batch + i) >= trainTotal * (1 + negRate))
            {
                std::cerr << "Error: batch index " << (batch + i) << " out of bounds" << std::endl;
                continue;
            }

            if (static_cast<float>(Random::randd(id) % 1000) < headCorruptProb[trainList[dataIdx].r])
            {
                batch_h[batch + i] = trainList[dataIdx % trainTotal].h;
                batch_r[batch + i] = trainList[dataIdx % trainTotal].r;
                batch_t[batch + i] = corrupt_tail(id, trainList[dataIdx].h, trainList[dataIdx].r);
                batch_y[batch + i] = -1;
            }
            else
            {
                batch_h[batch + i] = corrupt_head(id, trainList[dataIdx].t, trainList[dataIdx].r);
                batch_r[batch + i] = trainList[dataIdx % trainTotal].r;
                batch_t[batch + i] = distribution(generator);
                batch_y[batch + i] = -1;
            }
        }
    }

    return NULL;
}

void importTrainFiles()
{
    std::string filePath = inPath + "train2id.txt";
    std::ifstream fin(filePath);
    if (!fin.is_open())
    {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return;
    }

    fin >> trainTotal;
    if (fin.fail())
    {
        std::cerr << "Error reading train total from file: " << filePath << std::endl;
        return;
    }

    trainList = new Triple[trainTotal];
    trainHead = new Triple[trainTotal];
    trainTail = new Triple[trainTotal];
    std::vector<INT> freqRel(relationTotal, 0);

    for (int i = 0; i < trainTotal; ++i)
    {
        fin >> trainList[i].h >> trainList[i].t >> trainList[i].r;
        freqRel[trainList[i].r]++;
    }
    fin.close();

    std::memcpy(trainHead, trainList, sizeof(Triple) * trainTotal);
    std::memcpy(trainTail, trainList, sizeof(Triple) * trainTotal);

    std::sort(trainHead, trainHead + trainTotal, Triple::cmp_head);
    std::sort(trainTail, trainTail + trainTotal, Triple::cmp_tail);

    // writeTriplesToFile("mina.txt", trainHead, trainTotal);

    headStartIndices = new INT[entityTotal]();
    headEndIndices = new INT[entityTotal]();
    tailStartIndices = new INT[entityTotal]();
    tailEndIndices = new INT[entityTotal]();

    memset(headStartIndices, -1, sizeof(INT) * entityTotal);
    memset(tailStartIndices, -1, sizeof(INT) * entityTotal);

    memset(headEndIndices, -1, sizeof(INT) * entityTotal);
    memset(tailEndIndices, -1, sizeof(INT) * entityTotal);

    for (INT i = 1; i < trainTotal; i++)
    {
        if (trainHead[i].h != trainHead[i - 1].h)
        {
            headEndIndices[trainHead[i - 1].h] = i - 1;
            headStartIndices[trainHead[i].h] = i;
        }
        if (trainTail[i].t != trainTail[i - 1].t)
        {
            tailEndIndices[trainTail[i - 1].t] = i - 1;
            tailStartIndices[trainTail[i].t] = i;
        }
    }

    headStartIndices[trainHead[0].h] = 0;
    headEndIndices[trainHead[trainTotal - 1].h] = trainTotal - 1;

    tailStartIndices[trainTail[0].t] = 0;
    tailEndIndices[trainTail[trainTotal - 1].t] = trainTotal - 1;

    left_mean.resize(relationTotal, 0);
    right_mean.resize(relationTotal, 0);

    for (INT i = 0; i < entityTotal; i++)
    {
        int start = headStartIndices[i];
        int end = headEndIndices[i];

        if (start != -1 && end != -1)
        {
            for (INT j = start + 1; j <= end; j++)
            {
                if (trainHead[j].r != trainHead[j - 1].r)
                {
                    left_mean[trainHead[j].r] += 1.0;
                }
            }
            if (headStartIndices[i] <= headEndIndices[i])
            {
                left_mean[trainHead[headStartIndices[i]].r] += 1.0;
            }
        }
        start = tailStartIndices[i];
        end = tailEndIndices[i];
        if (start != -1 && end != -1)
        {
            for (INT j = start + 1; j <= end; j++)
            {
                if (trainTail[j].r != trainTail[j - 1].r)
                {
                    right_mean[trainTail[j].r] += 1.0;
                }
            }
            if (tailStartIndices[i] <= tailEndIndices[i])
            {
                right_mean[trainTail[tailStartIndices[i]].r] += 1.0;
            }
        }
    }

    headCorruptProb = new REAL[relationTotal];
    for (INT i = 0; i < relationTotal; i++)
    {
        left_mean[i] = freqRel[i] / left_mean[i];
        right_mean[i] = freqRel[i] / right_mean[i];
        headCorruptProb[i] = 1000 * right_mean[i] / (right_mean[i] + left_mean[i]);
    }

    std::cout << "Finished importing training data" << std::endl;
}

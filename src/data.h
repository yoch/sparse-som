#ifndef DATA_H_INCLUDED
#define DATA_H_INCLUDED



#include <vector>
#include <string>

#include "sparse_vec.h"


struct dataset
{
    std::vector<sparse_vec> samples;
    std::vector<std::string> labels;
    size_t nfeatures() const;
    size_t nsamples() const
    {
        return samples.size();
    }
};

// load data at SVM sparse format
dataset loadSparseData(const std::string& filename, uint32_t offset=0);


#endif // DATA_H_INCLUDED

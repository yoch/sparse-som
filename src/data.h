#ifndef DATA_H_INCLUDED
#define DATA_H_INCLUDED



#include <vector>
#include <string>


struct CSR
{
    float * data;
    int * indices;
    int * indptr;
    int nrows;
    int ncols;
    int nnz;

    void normalize();
};


class dataset : public CSR
{
public:

    dataset();
    // load from filename
    dataset(const std::string& filename, int offset=0);

    size_t nfeatures() const { return ncols; }
    size_t nsamples() const { return nrows; }

    std::vector<std::string> labels;

private:
    std::vector<float> _data;
    std::vector<int> _indices;
    std::vector<int> _indptr;
};


#endif // DATA_H_INCLUDED

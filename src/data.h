#ifndef DATA_H_INCLUDED
#define DATA_H_INCLUDED



#include <vector>
#include <string>


struct CSR
{
    float * data;
    int * indices;
    int * indptr;
    float * _sqsum;
    int nrows;
    int ncols;
    int nnz;

    void normalize();
};


class dataset : public CSR
{
public:

    dataset();
    dataset(const std::string& filename, int offset=0);     // load from filename
    dataset(const dataset&) = delete;                       // disable copy
    ~dataset()
    {
        if (_sqsum) delete [] _sqsum;
    }

    std::vector<std::string> labels;

private:
    std::vector<float> _data;
    std::vector<int> _indices;
    std::vector<int> _indptr;
};


#endif // DATA_H_INCLUDED

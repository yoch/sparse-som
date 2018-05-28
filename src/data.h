#ifndef DATA_H_INCLUDED
#define DATA_H_INCLUDED



#include <vector>
#include <string>


struct CSR
{
/*
    CSR()
    {}

    CSR(float * _data, int * _indices, int * _indptr, int _nrows, int _ncols, int _nnz) :
        data(_data),
        indices(_indices),
        indptr(_indptr),
        nrows(_nrows),
        ncols(_ncols),
        nnz(_nnz)
    {
        //initSqSum();
    }

    CSR(const CSR&) = delete;
    CSR& operator=(const CSR&) = delete;
*/
    ~CSR()
    {
        if (_sqsum) delete [] _sqsum;
    }

    void initSqSum();
    void normalize();

    float * data;
    int * indices;
    int * indptr;
    int nrows;
    int ncols;
    int nnz;
    float * _sqsum; // to store X^2
};


class dataset : public CSR
{
public:

    dataset();
    dataset(const std::string& filename, int offset=0);     // load from filename
    dataset(const dataset&) = delete;                       // disable copy
    dataset& operator=(const dataset&) = delete;            // disable assignment

    std::vector<std::string> labels;

private:
    std::vector<float> _data;
    std::vector<int> _indices;
    std::vector<int> _indptr;
};


#endif // DATA_H_INCLUDED

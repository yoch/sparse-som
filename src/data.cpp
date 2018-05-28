#include "data.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>

using namespace std;


void CSR::normalize()
{
    assert(_sqsum != NULL);

#pragma omp parallel for
    for (int i=0; i<nrows; ++i)
    {
        const float norm = std::sqrt(_sqsum[i]);
        for (int j=indptr[i]; j<indptr[i+1]; ++j)
        {
            data[j] /= norm;
        }
    }
}

void CSR::initSqSum()
{
    _sqsum = new float[nrows];

#pragma omp parallel for
    for (int i=0; i < nrows; ++i)
    {
        double sumOfSquares = 0.;
        for (int j=indptr[i]; j<indptr[i+1]; ++j)
        {
            sumOfSquares += data[j] * data[j];
        }
        _sqsum[i] = sumOfSquares;
    }
}


dataset::dataset()
{}

dataset::dataset(const string& filename, int offset)
{
    ifstream myfile;
    myfile.open(filename);

    if (!myfile.is_open())
        throw "impossible d'ouvrir le fichier " + string(filename);

    string line;
    size_t nline = 0;
    ncols = 0;  // store maximal indice found

    _indptr.push_back(0);

    while(!myfile.eof())
    {
        getline(myfile, line);
        ++nline;

        if (line.size()==0)
            continue;

        // skip comments
        if (line[0] == '#')
            continue;

        char * buf = (char*) line.c_str();
        char label[128];

/*
        // skip the label (if any)
        char * s = strchr(buf, ' ');
        if (s == NULL || strchr(buf, ':') < s)
            s = buf;
*/

        //TODO: make labels optional ?
        char * s = strchr(buf, ' ');
        if (!s || s==buf)
        {
            throw "no label found. at line: " + to_string(nline);
        }
        else if (s-buf >= (int)sizeof(label)-1)
        {
            throw "invalid label (too long). at line: " + to_string(nline);
        }
        strncpy(label, buf, s-buf);
        label[s-buf] = '\0';
        
        // store this label
        labels.push_back(label);

        // scan values
        while (*s != '#' && *s != '\n' && *s != '\0')
        {
            char * end;

            // get column index
            int idx = strtoul(s, &end, 10);
            if (end==s)
            {
                throw "bad index '" + string(s).substr(0, 8)
                        + "'. at line: " + to_string(nline) + ", col: " + to_string(s-buf+1);
            }
            if (offset > idx)
            {
                throw "bad index " + to_string(idx) + " for offset " + to_string(offset)
                        + ". at line: " + to_string(nline) + ", col: " + to_string(s-buf+1);
            }
            s = end;

            // verify separator
            while(*s==' ') ++s;
            if (*s!=':')
            {
                throw "bad separator '" + string(s, 1)
                        + "'. at line: " + to_string(nline) + ", col: " + to_string(s-buf+1);
            }
            ++s;

            // get value
            float val = strtof(s, &end);
            if (end==s)
            {
                throw "bad value '" + string(s).substr(0, 8)
                        + "'. at line: " + to_string(nline) + ", col: " + to_string(s-buf+1);
            }
            s = end;

            idx -= offset;
            if (idx > ncols)
            {
                ncols = idx;
            }

            // prevents unordered row
            if (_data.size() > (size_t) _indptr.back() && idx <= _indices.back())
            {
                throw "unordered or duplicate rows don't allowed."
                        " at line: " + to_string(nline) + ", col: " + to_string(s-buf+1);
            }

            _data.push_back(val);
            _indices.push_back(idx);

            // skip trailing whitespaces
            while(*s==' ') ++s;
        }

        _indptr.push_back(_data.size());
    }

    myfile.close();

    // store the pointers
    data = _data.data();
    indices = _indices.data();
    indptr = _indptr.data();

    // set additional values
    ++ncols;    // convert max indice to ncols
    nrows = _indptr.size() - 1;
    nnz = _data.size();


    // After all, init X^2
    initSqSum();

}

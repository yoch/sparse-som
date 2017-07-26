#include "data.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <numeric>

using namespace std;


dataset loadSparseData(const string& filename, uint32_t offset)
{
    dataset dataSet;

    ifstream myfile;
    myfile.open(filename);

    if (!myfile.is_open())
        throw "impossible d'ouvrir le fichier " + filename;

    string line;
    size_t nline = 0;

    while(!myfile.eof())
    {
        getline(myfile, line);
        ++nline;

        if (line.size()==0)
            continue;

        // skip comments
        if (line[0] == '#')
            continue;

        size_t nmemb = count(line.cbegin(), line.cend(), ':');
        if (nmemb == 0)
        {
            throw "invalid line: " + to_string(nline);
        }

        sparse_vec vec;
        vec.reserve(nmemb);

        char * buf = (char*) line.c_str();
        char label[128];

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

        while (*s != '#' && *s != '\n' && *s != '\0')
        {
            char * end;

            // get column index
            uint32_t idx = strtoul(s, &end, 10);
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

            // prevents unordered row
            if (vec.size() > 0 && idx-offset <= vec.back().idx)
            {
                throw "unordered or duplicate rows don't allowed."
                        " at line: " + to_string(nline) + ", col: " + to_string(s-buf+1);
            }

            vec.push_back(cell{idx-offset, val});

            // skip trailing whitespaces
            while(*s==' ') ++s;
        }

        dataSet.labels.push_back(label);
        dataSet.samples.push_back(move(vec));
    }

    myfile.close();

    return dataSet;
}

// assume vectors is sorted
size_t dataset::nfeatures() const
{
    unsigned maxi = 0;
    for (const sparse_vec& v : samples)
    {
        maxi = max(maxi, v.back().idx + 1);
    }
    return maxi;
}

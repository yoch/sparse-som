#include "../data.h"
#include "../som.h"


#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <cassert>
#include <cfloat>
#include <unistd.h> // getopt


using namespace std;


static void usage(const char* name)
{
    cerr << "Usage: " << name << endl
        << "\t-i infile - input file at libsvm sparse format" << endl
        << "\t-y nrows  - number of rows in the codebook" << endl
        << "\t-x ncols  - number of columns in the codebook" << endl
        << "\t[ -u ] - one based column indices (default is zero based)" << endl
        << "\t[ -N ] - normalize the input vectors" << endl
        << "\t[ -l codebook ]   - load codebook from binary file" << endl
        << "\t[ -o|O codebook ] - output codebook to filename (o:binary, O:text)" << endl
        << "\t[ -c|C classes ]  - output classification (c:without counts, C:with counts)" << endl
        << "\t[ -n neighborhood ] - neighborhood topology: 4=circ, 6=hexa, 8=rect (default 8)" << endl
        << "\t[ -T epochs ] - number of epochs (default 10)" << endl
        << "\t[ -r radius0 -R radiusN ] - radius at start and end (default r=(x+y)/2, R=0.5)" << endl
        << "\t[ -H radiusCool ] - radius cooling: 0=linear, 1=exponential (default 0)" << endl
        << "\t[ -s stdCoeff ]   - sigma = radius * coeff (default 0.3)" << endl
        << "\t[ -q ] - set verbosity level to 0 (default 1)" << endl
        << "\t[ -v ] - set verbosity level to 2 (default 1)" << endl;
    exit(-1);
}

int main(int argc, char *argv[])
{
    int x=-1, y=-1, ncols=-1, tcoef=-1, n;
    float r0=-1, rN=0.5, sc=0.3;
    som::topology neigh = som::RECT;
    som::cooling rcool = som::LINEAR;
    string filename, codebookfile, classfile, loadfile;
    char outType=0, classType=0;
    bool normalize = false, zerobased = true, load=false;
    int verbose=1;

    // Pase command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "i:o:O:l:C:c:x:y:t:T:r:R:a:A:s:n:Nuqv")) != -1)
    {
        switch (opt) {
        case 'i':   // infile
            filename = optarg;
            break;
        case 'u':
            zerobased = false;
            break;
        case 'o':   // dump codebook
        case 'O':
            outType = opt;
            codebookfile = optarg;
            break;
        case 'l':
            load = true;
            loadfile = optarg;
            break;
        case 'c':	// dump classification
        case 'C':
            classType = opt;
            classfile = optarg;
            break;
        case 'x':
            x = atoi(optarg);
            assert(x>0);
            break;
        case 'y':
            y = atoi(optarg);
            assert(y>0);
            break;
        case 'T':   // epochs
            tcoef = atoi(optarg);
            assert(tcoef>=0);
            break;
        case 'n':   // neighborhood
            n = atoi(optarg);
            assert(n==4 || n==6 || n==8);
            neigh = static_cast<som::topology>(n);
            break;
        case 'r':   // radius0
            r0 = atof(optarg);
            assert(r0>0);
            break;
        case 'R':   // radiusN
            rN = atof(optarg);
            assert(rN>0);
            break;
/*
        case 'a':   // alpha0
            a0 = atof(optarg);
            assert(a0>0 && a0<=1);
            break;
        case 'A':   // alphaN
            aN = atof(optarg);
            assert(aN>0 && aN<1);
            break;
*/
        case 's':   // std coeff
            sc = atof(optarg);
            assert(sc >= 0.05 && sc <= 1);
            break;
        case 'N':
            normalize = true;
            break;
        case 'q':   // quiet
            verbose = 0;
            break;
        case 'v':   // verbose
            verbose = 2;
            break;
        default: /* '?' */
            usage(argv[0]);
        }
    }

    if ((x==-1 || y==-1) && !load)
    {
        usage(argv[0]);
    }

    if (r0 != -1 && r0 <= rN)
    {
        cerr << "bad parameters" << endl;
        return -1;
    }

    dataset dataSet;
    try
    {
        if (verbose > 0)
        {
            cout << "Loading dataset... ";
            cout.flush();
        }

        clock_t tm1 = clock();
        dataSet = loadSparseData(filename, zerobased ? 0 : 1);
        clock_t tm2 = clock();
        ncols = dataSet.nfeatures();

        if (verbose > 0)
        {
            cout << "OK (" << (float)(tm2 - tm1) / CLOCKS_PER_SEC << "s)" << endl;
            if (verbose > 1)
            {
                cout << "  " << dataSet.nsamples() << " vectors read" << endl;
                cout << "  " << ncols << " features" << endl;
            }
        }

        // normalize vectors
        if (normalize)
        {
            if (verbose > 0)
            {
                cout << "Normalize the dataset... ";
                cout.flush();
            }
            for (sparse_vec& v: dataSet.samples)
            {
                v.normalize();
            }
            if (verbose > 0)
            {
                cout << "OK" << endl;
            }
        }
    }
    catch(string& err)
    {
        cerr << err << endl;
        return -1;
    }

    if (tcoef==-1)
    {
        tcoef = 10;
    }

    som::BSom som = load ?
                som::BSom(loadfile, neigh, verbose) :
                som::BSom(y, x, ncols, neigh, verbose);

    if (r0==-1)
    {
        r0 = (float) (som.getx() + som.gety()) / 2;
    }

    som.train(dataSet.samples, tcoef, r0, rN, sc, rcool);

    // codebook output
    if (!codebookfile.empty())
    {
        if (verbose > 0)
        {
            cout << " writing codebook to " << codebookfile << " ...";
            cout.flush();
        }

        try
        {
            if (outType == 'o')
            {
                som.dump(codebookfile, true);
            }
            if (outType == 'O')
            {
                som.dump(codebookfile, false);
            }
        }
        catch(string& err)
        {
            cerr << err << endl;
            return -1;
        }
        
        if (verbose > 0)
        {
            cout << " OK" << endl;
        }
    }

    // classification output
    if (!classfile.empty())
    {
        if (verbose > 0)
        {
            cout << " writing classification to " << classfile << " ...";
            cout.flush();
        }

        bool count = (classType=='C');
        try
        {
            som.classify(dataSet, classfile, count);
        }
        catch(string& err)
        {
            cerr << err << endl;
            return -1;
        }
        if (verbose > 0)
        {
            cout << " OK" << endl;
        }
    }

    return 0 ;
}

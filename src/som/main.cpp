#include "../data.h"
#include "../som.h"

#include <cstdlib>
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
        << "\t[ -t tmax | -T epochs ]   - number of training iterations (epoch=nb. of samples)" << endl
        << "\t[ -r radius0 -R radiusN ] - radius at start and end (default r=(x+y)/2, R=0.5)" << endl
        << "\t[ -a alpha0  -A  alphaN ] - learning rate at start and end (default a=0.5, A=1.e-37)" << endl
        << "\t[ -H radiusCool ] - radius cooling: 0=linear, 1=exponential (default 0)" << endl
        << "\t[ -h  alphaCool ] - alpha cooling: 0=linear, 1=exponential (default 0)" << endl
        << "\t[ -s stdCoeff ]   - sigma = radius * coeff (default 0.3)" << endl
        << "\t[ -q ] - quiet" << endl;
    exit(-1);
}

int main(int argc, char *argv[])
{
    int x=-1, y=-1, ncols=-1, tmax=-1, tcoef=-1, n;
    double r0=-1, rN=FLT_MIN, a0=0.5, aN=FLT_MIN, sc=0.3;
    som::topology neigh = som::RECT;
    som::cooling acool=som::LINEAR, rcool=som::LINEAR;
    string filename, codebookfile, classfile, loadfile;
    char outType=0, classType=0;
    bool normalize = false, zerobased = true, verbose = true, load=false;

    // Pase command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "i:o:O:l:C:c:x:y:t:T:r:R:a:A:s:n:h:H:Nuq")) != -1)
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
        case 't':   // Tmax
            assert(tcoef==-1);
            tmax = atoi(optarg);
            assert(tmax>0);
            break;
        case 'T':   // epochs
            assert(tmax==-1);
            tcoef = atoi(optarg);
            assert(tcoef>0);
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
        case 'a':   // alpha0
            a0 = atof(optarg);
            assert(a0>0 && a0<=1);
            break;
        case 'A':   // alphaN
            aN = atof(optarg);
            assert(aN>0 && aN<1);
            break;
        case 's':   // std coeff
            sc = atof(optarg);
            assert(sc >= 0.05 && sc <= 1);
            break;
        case 'H':   // radius cooling
            n = atoi(optarg);
            assert(n==0 || n==1);
            rcool = static_cast<som::cooling>(n);
            break;
        case 'h':   // alpha cooling
            n = atoi(optarg);
            assert(n==0 || n==1);
            acool = static_cast<som::cooling>(n);
            break;
        case 'N':
            normalize = true;
            break;
        case 'q':   // quiet
            verbose = false;
            break;
        default: /* '?' */
            usage(argv[0]);
        }
    }

    if ((x==-1 || y==-1) && !load)
    {
        usage(argv[0]);
    }

    if ((r0 != -1 && r0 <= rN) || (a0 <= aN))
    {
        cerr << "bad parameters" << endl;
        return -1;
    }

    dataset dataSet;
    try
    {
        if (verbose)
        {
            cout << "Loading dataset... ";
            cout.flush();
        }

        dataSet = loadSparseData(filename, zerobased ? 0 : 1);
        ncols = dataSet.nfeatures();

        if (verbose)
        {
            cout << "OK" << endl;
            cout << "  " << dataSet.nsamples() << " vectors read" << endl;
            cout << "  " << ncols << " features" << endl;
        }

        // normalize vectors
        if (normalize)
        {
            for (sparse_vec& v: dataSet.samples)
            {
                v.normalize();
            }
        }
    }
    catch(string& err)
    {
        cerr << err << endl;
        return -1;
    }

    if (tmax==-1)
    {
        tmax = ((tcoef==-1) ? 10 : tcoef) * dataSet.samples.size();
    }


    som::Som som = load ?
                som::Som(loadfile, neigh, verbose) :
                som::Som(y, x, ncols, neigh, verbose);

    if (r0==-1)
    {
        r0 = (float) (som.getx() + som.gety()) / 2;
    }

    som.train(dataSet.samples, tmax, r0, a0, rN, aN, sc, rcool, acool);

    // codebook output
    if (!codebookfile.empty())
    {
        if (verbose)
        {
            cout << " writing codebook to " << codebookfile << endl;
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
    }

    // classification output
    if (!classfile.empty())
    {
        if (verbose)
        {
            cout << " writing classification to " << classfile << endl;
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
    }

    return 0 ;
}

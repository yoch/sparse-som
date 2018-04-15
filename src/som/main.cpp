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
        << "\t-i infile        input file at libsvm sparse format" << endl
        << "\t-y nrows         number of rows in the codebook" << endl
        << "\t-x ncols         number of columns in the codebook" << endl
        << "\t[ -d dim ]       force the dimension of codebook's vectors" << endl
        << "\t[ -u ]           one based column indices (default is zero based)" << endl
        << "\t[ -N ]           normalize the input vectors" << endl
        << "\t[ -l cb ]        load codebook from binary file" << endl
        << "\t[ -o|O cb ]      output codebook to filename (o:binary, O:text)" << endl
        << "\t[ -c|C cl ]      output classification (c:without counts, C:with counts)" << endl
        << "\t[ -n neig ]      neighborhood topology: 4=circ, 6=hexa, 8=rect (default 8)" << endl
        << "\t[ -t n | -T e ]  number of training iterations or epochs (epoch = nrows)" << endl
        << "\t[ -r r0 -R rN ]  radius at start and end (default r=(x+y)/2, R=0.5)" << endl
        << "\t[ -a a0 -A aN ]  learning rate at start and end (default a=0.5, A=1.e-37)" << endl
        << "\t[ -H rCool ]     radius cooling: 0=linear, 1=exponential (default 0)" << endl
        << "\t[ -h aCool ]     alpha cooling: 0=linear, 1=exponential (default 0)" << endl
        << "\t[ -s stdCf ]     sigma = radius * stdCf (default 0.3)" << endl
        << "\t[ -v ]           increase verbosity level (default 0, max 2)" << endl;
    exit(-1);
}

int main(int argc, char *argv[])
{
    int x=-1, y=-1, d=-1, tmax=-1, tcoef=-1, n;
    double r0=-1, rN=FLT_MIN, a0=0.5, aN=FLT_MIN, sc=0.3;
    som::topology neigh = som::RECT;
    som::cooling acool=som::LINEAR, rcool=som::LINEAR;
    string filename, codebookfile, classfile, loadfile;
    char outType=0, classType=0;
    bool normalize = false, zerobased = true, load=false;
    int verbose=0;
    double wtime=0;

    // Pase command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "i:o:O:l:C:c:x:y:d:t:T:r:R:a:A:s:n:h:H:Nuv")) != -1)
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
        case 'd':
            d = atoi(optarg);
            assert(d>0);
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
        case 'v':   // verbose
            verbose++;
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

    if (verbose > 0)
    {
        cout << "Loading dataset... ";
        cout.flush();

        wtime = get_wall_time();
    }

    dataset dataSet(filename, zerobased ? 0 : 1);
    assert(d == -1 || d >= dataSet.ncols);

/*
    catch(string& err)
    {
        //cout << endl;
        cerr << err << endl;
        return -1;
    }
*/

    if (verbose > 0)
    {
        wtime = get_wall_time() - wtime;

        cout << "OK (" << wtime << "s)" << endl;
        if (verbose > 1)
        {
            cout << "  " << dataSet.nrows << " rows" << endl;
            cout << "  " << dataSet.ncols << " cols" << endl;
            cout << "  " << dataSet.nnz << " nnz" << endl;
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
        dataSet.normalize();
        if (verbose > 0)
        {
            cout << "OK" << endl;
        }
    }

    if (tmax==-1)
    {
        tmax = ((tcoef==-1) ? 10 : tcoef) * dataSet.nrows;
    }


    som::Som som = load ?
                som::Som(loadfile, neigh, verbose) :
                som::Som(y, x, d == -1 ? dataSet.ncols : d, neigh, verbose);

    if (r0==-1)
    {
        r0 = (float) (som.getx() + som.gety()) / 2;
    }

    som.train(dataSet, tmax, r0, a0, rN, aN, sc, rcool, acool);

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
            //cout << endl;
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

        try
        {
            bool count = (classType=='C');
            som.classify(dataSet, classfile, count);
        }
        catch(string& err)
        {
            //cout << endl;
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

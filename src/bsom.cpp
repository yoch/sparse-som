#include "som.h"

#include <random>
#include <ctime>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <sstream>

#if defined(_OPENMP)
    #include <omp.h>
#endif

using namespace std;
using namespace som;

#if defined(_MSC_VER)
    using idx_t = int;
#else
    using idx_t = unsigned int;
#endif


//#define HEX_H 0.8660254037844386

//size_t SKIPPED = 0;


BSom::BSom(size_t h, size_t w, size_t d, topology topo, int verbose) :
    m_height(h),
    m_width(w),
    m_dim(d),
    m_topo(topo),
    m_verbose(verbose)
{
    setTopology(topo);

    codebook = new float[h*w*d];

    init();
}

BSom::BSom(const std::string& filename, topology topo, int verbose) :
    m_topo(topo),
    m_verbose(verbose)
{
    setTopology(topo);

    ifstream myfile;
    myfile.open(filename, ios::binary);

    if (!myfile.is_open())
        throw "unable to open " + filename;

    if (m_verbose > 0)
    {
        cout << "loading codebook...";
        cout.flush();
    }

    string header;
    getline(myfile, header);

    istringstream iss(header);
    iss >> m_height >> m_width >> m_dim;

    if (iss.fail())
        throw "bad header in " + filename;

    // allocate memory
    codebook = new float[m_height * m_width * m_dim];

    myfile.read((char*)codebook, m_height * m_width * m_dim * sizeof(*codebook));

    if (myfile.fail())
        throw "bad header in " + filename;

    myfile.close();

    if (m_verbose > 0)
    {
        cout << " OK" << endl;
        if (m_verbose > 1)
        {
            cout << "  dimensions: " << m_height << " x " << m_width << " x " << m_dim << endl;
        }
    }
}

BSom::~BSom()
{
    delete [] codebook;
}

void BSom::setTopology(topology topo)
{
    m_topo = topo;

    switch (m_topo)
    {
    case CIRC:
        fdist = eucdist;
        break;
    case HEXA:
        fdist = eucdist_hexa;
        break;
    case RECT:
    default:
        fdist = rectdist;
        break;
    }
}

static float prod(const float * const vsp, const int * const vind, const size_t vsz, const float * const w)
{
    float ret = 0.;
    for (size_t it=0; it<vsz; ++it)
    {
        ret += vsp[it] * w[vind[it]];
    }
    return ret;
}

static float squared(const float * const w, size_t sz)
{
    float ret = 0.;
    for (size_t i=0; i<sz; ++i)
    {
        ret += squared(w[i]);
    }
    return ret;
}

/*
static inline float euclideanDistanceSq(const float * const vsp, const int * const vind, const size_t vsz, const float * const w, float w2)
{
    return max(0.f, w2 - 2 * prod(vsp, vind, vsz, w) + x2);
}
*/

/// init codebook at random
void BSom::init()
{
    unsigned seed = time(NULL);
#pragma omp parallel firstprivate(seed)
  {
#if defined(_OPENMP)
    // set an unique seed for each thread
    seed += omp_get_thread_num();
#endif
    default_random_engine rng(seed);
    uniform_real_distribution<float> dist(0., 1.);

#pragma omp for
    for (idx_t k=0; k < m_height*m_width*m_dim; ++k)
    {
        codebook[k] = dist(rng);
    }
  }
}

void BSom::getBmus(const CSR& data, size_t * const bmus, float * const dsts, size_t * const second, bool correct) const
{
    assert(data._sqsum != NULL || !correct);

    float * sdsts = NULL;

    fill_n(bmus, data.nrows, 0);
    fill_n(dsts, data.nrows, FLT_MAX);

    if (second)
    {
        sdsts = new float[data.nrows];;
        fill_n(second, data.nrows, 0);
        fill_n(sdsts, data.nrows, FLT_MAX);
    }

    for (size_t k=0; k < m_height * m_width; ++k)
    {
        const float * const w = codebook + k * m_dim;

        // precompute {w_k}^2
        const float w2 = squared(w, m_dim);

// ensure that two threads don't access shared data (at i index) at a time
#pragma omp parallel for // shared(data,k,w2)

        for (idx_t i=0; i < (idx_t) data.nrows; ++i)
        {
            const int ind = data.indptr[i];
            const int vsz = data.indptr[i+1] - ind;
            const float * const vsp = &data.data[ind];
            const int * const vind = &data.indices[ind];

            // pseudo squared distance, d_i = d_i + X_i^2
            const float dst = w2 - 2 * prod(vsp, vind, vsz, w);

            if (!second)
            {
                if (dst < dsts[i])
                {
                    bmus[i] = k;
                    dsts[i] = dst;
                }
            }
            else // second!=NULL
            {
                if (dst < dsts[i])
                {
                    second[i] = bmus[i];
                    sdsts[i] = dsts[i];
                    bmus[i] = k;
                    dsts[i] = dst;
                }
                else if (dst < sdsts[i])
                {
                    second[i] = k;
                    sdsts[i] = dst;
                }
            }
        }
    }

    if (correct)
    {
#pragma omp parallel for
        for (idx_t i=0; i < (idx_t) data.nrows; ++i)
        {
            dsts[i] = max(0.f, dsts[i] + data._sqsum[i]);
        }
    }

    if (sdsts) delete [] sdsts;
}

void BSom::update(const CSR& data, const float radius, const float stdCoef, size_t * const bmus)
{
//    SKIPPED = 0;
    const double gamma = -1. / (2. * squared(radius * stdCoef));

#pragma omp parallel // shared(data)
 {
    float * const numerator = new float[m_dim];

#pragma omp for

    for (idx_t xy=0; xy < m_height * m_width; ++xy)
    {
        const int y = xy / m_width,
                  x = xy % m_width;

        // init numerator and denominator
        float denominator = 0.f;
        fill_n(numerator, m_dim, 0.f);

        for (size_t n=0; n < (size_t) data.nrows; ++n)
        {
            const int ind = data.indptr[n];
            const int vsz = data.indptr[n+1] - ind;
            const float * const vsp = &data.data[ind];
            const int * const vind = &data.indices[ind];

            const int i = bmus[n] / m_width,
                      j = bmus[n] % m_width;

            double d2;

            if (fdist(y, x, i, j, d2) > radius+1)
                continue;

            const float h = exp(gamma * d2);

            denominator += h;
            for (int it=0; it<vsz; ++it)
            {
                numerator[vind[it]] += h * vsp[it];
            }
        }

        if (denominator != 0)
        {
            float * const w = codebook + xy * m_dim;
            for (size_t k=0; k < m_dim; ++k)
            {
                w[k] = numerator[k] / denominator;
            }
        }
/*
        else
        {
#pragma omp atomic
            SKIPPED++;
        }
*/
    }
    delete [] numerator;
 }
}

double BSom::topographicError(size_t * const bmus, size_t * const second, size_t n) const
{
    size_t errors = 0;
#pragma omp parallel for reduction(+: errors)
    for (size_t k=0; k<n; ++k)
    {
        double d2;
        const int y0 = bmus[k] / m_width,
                  x0 = bmus[k] % m_width,
                  y1 = second[k] / m_width,
                  x1 = second[k] % m_width;
        if (fdist(y0, x0, y1, x1, d2) > 1)
        {
            errors++;
        }
    }
    //cout << endl << errors << " topographic errors on " << n << " samples" << endl;
    return (double) errors / n;
}

void BSom::trainOneEpoch(const CSR& data, float radius, float stdCoef,
                        size_t * const bmus, float * const dsts)
{

    size_t * second = NULL;
    bool correct = false;

    if (m_verbose > 1)
    {
        second = new size_t[data.nrows];
        if (data._sqsum)
            correct = true;
    }

    ///////////////////////////////////////////
    ///          Searching BMUs
    getBmus(data, bmus, dsts, second, correct);

    //////////////////////////////////////////
    ///            Update phase
    update(data, radius, stdCoef, bmus);

    if (m_verbose > 1)
    {
        if (correct)
        {
            double Qe = 0;
#pragma omp parallel for reduction(+: Qe)
            for (int i=0; i<data.nrows; ++i)
            {
                Qe += sqrt(dsts[i]);
            }
            Qe /= data.nrows;

            // Note: in fact, this is the quantization error of the previous step (before the update)
            cout << " - QE: " << Qe;
        }

        if (second)
        {
            cout << " - TE: " << topographicError(bmus, second, data.nrows);
        }

        //cout << "  (" << SKIPPED << " units skipped)";
        // Note: in fact, this is the topographic error of the previous step (before the update)
        cout << endl;
    }

    if (second)
        delete [] second;
}

void BSom::train(const CSR& data, size_t tmax,
           float radius0, float radiusN, float stdCoef, cooling rcool)
{
    double tm = 0.;
    float radius;

    if (m_verbose > 0)
    {
        cout << "Start learning..." << endl;

        tm = get_wall_time();
    }

    size_t * bmus = new size_t[data.nrows];
    float * dsts = new float[data.nrows];

    for (size_t t=1; t<=tmax; ++t)
    {
        const float ratio = (double)(t-1) / (tmax-1);

        switch (rcool) {
        case EXPONENTIAL:
            radius = radius0 * powf(radiusN / radius0, ratio);
            break;
        case LINEAR:
        default:
            radius = radiusN + (radius0 - radiusN) * (1. - ratio);
            break;
        }

        trainOneEpoch(data, radius, stdCoef, bmus, dsts);

        if (m_verbose > 1)
        {
            cout << "  epoch " << t << " / " << tmax;
            cout << ", r = " << radius;
        }
    }

    if (m_verbose > 0)
    {

/* DBG
        if (m_verbose > 1)
        {
            size_t * second = new size_t[data.nrows];
            float * sdsts = new float[data.nrows];

            getBmus(data, bmus, dsts, second, sdsts);

            // unable to compute QE if we don't have X^2
            if (data._sqsum)
            {
                double Qe = 0;
                for (int i=0; i<data.nrows; ++i)
                {
                    Qe += sqrt(max(0.f, dsts[i] + data._sqsum[i]));
                }
                Qe /= data.nrows;

                // Note: in fact, this is the quantization error of the previous step (before the update)
                cout << " - QE: " << Qe;
            }

            if (second)
            {
                // Note: in fact, this is the topographic error of the previous step (before the update)
                cout << " - TE: " << topographicError(bmus, second, data.nrows);
            }

            delete [] second;
            delete [] sdsts;

            cout << endl;
        }
*/

        tm = get_wall_time() - tm;

        cout << "Finished: elapsed " << tm << "s" << endl;
    }

    delete [] bmus;
    delete [] dsts;
}

void BSom::dump(const std::string& filename, bool binary) const
{
    ofstream myfile;
    myfile.open(filename, ios::binary);

    if (!myfile.is_open())
        throw "unable to open " + filename;

    myfile << m_height << ' ' << m_width << ' ' << m_dim << '\n';

    if (binary)
    {
        myfile.write((const char*) codebook, m_height * m_width * m_dim * sizeof(*codebook));
    }
    else
    {
        for (size_t k=0; k < m_height*m_width; ++k)
        {
            float * w = codebook + k * m_dim;
            for (size_t i=0; i < m_dim; ++i)
                myfile << w[i] << ' ';
            myfile << '\n';
        }
    }

    myfile.close();
}

/*
void BSom::dumpSparse(const string& filename, double epsilon) const
{
    ofstream myfile;
    myfile.open(filename);

    if (!myfile.is_open())
        throw "unable to open " + filename;

    size_t cnt=0;

    //myfile << '#' << m_height << ' ' << m_width << ' ' << m_dim << endl;
    for (size_t k=0; k < m_height * m_width; ++k)
    {
        float * w = codebook + k * m_dim;
        for (size_t i=0; i < m_dim; ++i)
        {
            if (w[i] > epsilon)
            {
                myfile << i << ':' << w[i] << ' ';
                ++cnt;
            }
        }
        myfile << '\n';
    }

    myfile.close();
    cout << cnt << " values written" << endl;
}
*/

vector<label_counter> BSom::calibrate(const dataset& dataSet) const
{
    size_t * bmus = new size_t[dataSet.nrows];
    float * dsts = new float[dataSet.nrows];

    getBmus(dataSet, bmus, dsts);

    vector<label_counter> mappings = vector<label_counter>(m_height*m_width);
    for (int k=0; k < dataSet.nrows; ++k)
    {
        const size_t kStar = bmus[k];
        mappings[kStar][dataSet.labels[k]]++;
    }

    delete [] bmus;
    delete [] dsts;

    return mappings;
}

void BSom::classify(const dataset& dataSet, const string& filename, bool cnt) const
{
    ofstream myfile;
    myfile.open(filename);

    if (!myfile.is_open())
        throw "unable to open " + filename;

    const vector<label_counter> mappings = calibrate(dataSet);

    myfile << "CLASSES = {\n";
    for (size_t k=0; k < mappings.size(); ++k)
    {
        size_t y = k / m_width, x = k % m_width;
        myfile << " (" << y << ',' << x << "): { ";
        for (auto it = mappings[k].begin(); it != mappings[k].end(); ++it)
        {
            myfile << it->first;
            if (cnt)
            {
                myfile << ':' << it->second;
            }
            myfile << ", ";
        }
        myfile << "},\n";
    }
    myfile << "}\n";

    myfile.close();
}

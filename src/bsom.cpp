#include "som.h"

#include <random>
#include <ctime>
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


#define HEX_H 0.8660254037844386


BSom::BSom(size_t h, size_t w, size_t d, topology topo, bool verbose) :
    m_height(h),
    m_width(w),
    m_dim(d),
    m_topo(topo),
    m_verbose(verbose)
{
    codebook = new float[h*w*d];
    init();
}

BSom::BSom(const std::string& filename, topology topo, bool verbose) :
    m_topo(topo),
    m_verbose(verbose)
{
    ifstream myfile;
    myfile.open(filename, ios::binary);

    if (!myfile.is_open())
        throw "unable to open " + filename;

    if (m_verbose)
    {
        cout << "loading codebook..." << endl;
    }

    string header;
    getline(myfile, header);

    istringstream iss(header);
    iss >> m_height >> m_width >> m_dim;

    if (iss.fail())
        throw "bad header in " + filename;

    if (m_verbose)
    {
        cout << "  dimensions: " << m_height << "x" << m_width << "x" << m_dim << endl;
    }

    // allocate memory
    codebook = new float[m_height * m_width * m_dim];

    myfile.read((char*)codebook, m_height * m_width * m_dim * sizeof(*codebook));

    if (myfile.fail())
        throw "bad header in " + filename;

    myfile.close();

    if (m_verbose)
    {
        cout << "OK" << endl;
    }
}

BSom::~BSom()
{
    delete [] codebook;
}


static float prod(const sparse_vec& v, const float * const w)
{
    float ret = 0.;
    for (auto it=v.cbegin(); it!=v.cend(); ++it)
    {
        ret += it->val * w[it->idx];
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

static inline float euclideanDistanceSq(const sparse_vec& v, const float * const w, float w2)
{
    return max(0.f, w2 - 2 * prod(v, w) + v.sumOfSquares);
}

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

void BSom::getBmus(const vector<sparse_vec>& data, size_t * const bmus, float * const dsts) const
{
    fill_n(bmus, data.size(), 0);
    fill_n(dsts, data.size(), FLT_MAX);

    for (size_t k=0; k < m_height * m_width; ++k)
    {
        const float * const w = codebook + k * m_dim;

        // precompute {w_k}^2
        const float w2 = squared(w, m_dim);

// ensure that two threads don't access shared data (at i index) at a time
#pragma omp parallel for // shared(data,k)

        for (idx_t i=0; i < data.size(); ++i)
        {
            const float dst = euclideanDistanceSq(data[i], w, w2);

            if (dst < dsts[i])
            {
                bmus[i] = k;
                dsts[i] = dst;
            }
        }
    }
}

void BSom::update(const vector<sparse_vec>& data, const float radius, const float stdCoef, size_t * const bmus)
{
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

        for (size_t n=0; n < data.size(); ++n)
        {
            const sparse_vec& v = data[n];
            const int i = bmus[n] / m_width,
                      j = bmus[n] % m_width;

            bool so_far = true;
            float d2, x6, j6;
            switch (m_topo)
            {
            case CIRC:
                d2 = squared(i-y) + squared(j-x);
                so_far = d2 > squared(radius+1);
                break;
            case HEXA:
                x6 = (x&1) ? 0.5 + x : x;
                j6 = (j&1) ? 0.5 + j : j;
                d2 = squared((i-y)*HEX_H) + squared(j6-x6);
                so_far = d2 > squared(radius+1);
                break;
            case RECT:
            default:
                d2 = squared(i-y) + squared(j-x);
                so_far = max(abs(i-y),(j-x)) > radius+1;
                break;
            }

            if (so_far)
                continue;

            const float h = exp(-d2 / (2 * squared(radius * stdCoef)));

            denominator += h;
            for (auto it=v.cbegin(); it!=v.cend(); ++it)
            {
                numerator[it->idx] += h * it->val;
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
    }
    delete [] numerator;
 }
}

void BSom::trainOneEpoch(const vector<sparse_vec>& data, size_t t, size_t tmax,
                        float radius0, float radiusN, float stdCoef, cooling rcool,
                        size_t * const bmus, float * const dsts)
{
    float radius;
    const float ratio = ((float)t / tmax);

    switch (rcool) {
    case EXPONENTIAL:
        radius = radius0 * pow(radiusN / radius0, ratio);
        break;
    case LINEAR:
    default:
        radius = radiusN + (radius0 - radiusN) * (1. - ratio);
        break;
    }

    ///////////////////////////////////////////
    ///          Searching BMUs
    getBmus(data, bmus, dsts);

    //////////////////////////////////////////
    ///            Update phase
    update(data, radius, stdCoef, bmus);

    if (m_verbose)
    {
        float Qe = accumulate(dsts, dsts+data.size(), 0.f, [](float acc, float val){ return acc + sqrt(val); })
                        / data.size();
        cout << "  epoch " << t << " / " << tmax << " - QE " << Qe << endl;
    }
}

void BSom::train(const vector<sparse_vec>& data, size_t tmax,
           float radius0, float radiusN, float stdCoef, cooling rcool)
{
    clock_t tm1 = clock();

    if (m_verbose)
    {
        cout << "Start learning..." << endl;
    }

    size_t * bmus = new size_t[data.size()];
    float * dsts = new float[data.size()];

    for (size_t t=0; t<=tmax; ++t)
    {
        trainOneEpoch(data, t, tmax, radius0, radiusN, stdCoef, rcool, bmus, dsts);
    }

    delete [] bmus;
    delete [] dsts;

    clock_t tm2 = clock();

    if (m_verbose)
    {
        cout << "Finished: elapsed " << (float)(tm2 - tm1) / CLOCKS_PER_SEC << "s" << endl;
/*
        getBmus(data, bmus, dsts);
        float Qe = accumulate(dsts, dsts+data.size(), 0.f, [](float acc, float val){ return acc + sqrt(max(val,0.f)); })
                        / data.size();
        cout << "Quantization Error " << Qe << endl;
*/
    }
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
    size_t * bmus = new size_t[dataSet.nsamples()];
    float * dsts = new float[dataSet.nsamples()];

    getBmus(dataSet.samples, bmus, dsts);

    vector<label_counter> mappings = vector<label_counter>(m_height*m_width);
    for (size_t k=0; k < dataSet.nsamples(); ++k)
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

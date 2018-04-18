#include "som.h"

#include <cstdlib>
#include <random>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cfloat>
#include <cmath>
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


size_t NBSTABLES = 0;
#define EPSILON 1.e-50

//#define HEX_H 0.8660254037844386
#define SQRT2 1.4142135623730951


#define BUFSIZE 16777216
// IO buffer (64 Mo)
static float iobuf[BUFSIZE];


Som::Som(size_t h, size_t w, size_t d, topology topo, int verbose) :
    m_height(h),
    m_width(w),
    m_dim(d),
    m_topo(topo),
    m_verbose(verbose)
{
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
        fdist = manhatan_dist;
        break;
    }

    codebook = new double[h*w*d];

    // randomly init values
    init();
}

Som::Som(const std::string& filename, topology topo, int verbose) :
    m_topo(topo),
    m_verbose(verbose)
{
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
        fdist = manhatan_dist;
        break;
    }

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

    const size_t sz = m_height * m_width * m_dim;

    // allocate memory
    codebook = new double[sz];

    for (size_t k=0; k < sz; k+=BUFSIZE)
    {
        size_t toread = min<size_t>(BUFSIZE, sz - k);
        myfile.read((char*)iobuf, sz * sizeof(*iobuf));
        copy(iobuf, iobuf + toread, codebook + k);
    }

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

Som::~Som()
{
    delete [] codebook;
}


static double prod(const float * const vsp, const int * const ind, const size_t vsz, const double * const w)
{
    double ret = 0.;
    for (size_t it=0; it<vsz; ++it)
    {
        ret += vsp[it] * w[ind[it]];
    }
    return ret;
}

static double squared(const double * const w, size_t sz)
{
    double ret = 0.;
    for (size_t i=0; i < sz; ++i)
    {
        ret += squared(w[i]);
    }
    return ret;
}

static inline double euclideanDistanceSq(const float * const vsp, const int * const ind, const size_t vsz, const double x2, const double * const w, const double w2, const double wcoeff)
{
    return max(0., w2 - 2 * wcoeff * prod(vsp, ind, vsz, w) + x2);
}

static inline void partial_update(const float * const vsp, const int * const ind, const size_t n, double * const w, double vcoeff)
{
    for (size_t it=0; it<n; ++it)
    {
        w[ind[it]] += vsp[it] * vcoeff;
    }
}

void Som::init()
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

void Som::stabilize(size_t k)
{
    const double coeff = w_coeff[k];
    double * const w = codebook + k * m_dim;
    for (size_t i=0; i < m_dim; ++i)
    {
        w[i] *= coeff;
    }
    w_coeff[k] = 1.;
#pragma omp atomic
    ++NBSTABLES;
}


size_t Som::getBmu(const CSR& data, const size_t n, double& dStar) const
{
    const size_t ind = data.indptr[n];
    const size_t vsz = data.indptr[n+1] - ind;
    const float * const vsp = &data.data[ind];
    const int * const vind = &data.indices[ind];

    bmu mini = bmu(-1, DBL_MAX);
/*
#pragma omp declare \
    reduction(BMU : bmu : omp_out = omp_in.dst <= omp_out.dst ? omp_in : omp_out) \
    initializer (omp_priv(omp_orig))

#pragma omp parallel for \
    reduction(BMU:mini) \
    schedule(static,1)
*/
    for (size_t k=0; k < m_height*m_width; ++k)
    {
        //precompute w*v (to be used at update stage)
        wvprod[k] = prod(vsp, vind, vsz, codebook + k * m_dim);

        // fast squared euclidean dst (invariant by v^2) : w^2 - 2 w.v
        const double dst = squared_sum[k] - 2 * (wvprod[k] * w_coeff[k]);
        if (dst < mini.dst)
        {
            mini.idx = k;
            mini.dst = dst;
        }
    }

    // store values
    dStar = max(0., mini.dst + data._sqsum[n]); // correct euclidean dist
    return mini.idx;
}

void Som::update(const CSR& data, size_t n, size_t kStar, double radius, double alpha, double stdCoeff)
{
    const int x = kStar % m_width;
    const int y = kStar / m_width;
    const int r = radius;
    const int startI = max(0,y-r),
              stopI = min((int)m_height-1,y+r),
              startJ = max(0,x-r),
              stopJ = min((int)m_width-1,x+r);

    const int ind = data.indptr[n];
    const int vsz = data.indptr[n+1] - ind;
    const float * const vsp = &data.data[ind];
    const int * const vind = &data.indices[ind];

//#pragma omp parallel for collapse(2)
    //schedule(static,stopJ-startJ+1)
    //if ((stopI-startI)*(stopJ-startJ)>=100)
    for(int i = startI; i <= stopI; ++i)
    {
        for(int j = startJ; j <= stopJ; ++j)
        {
            double d2;

            if (fdist(y, x, i, j, d2) > radius+1)
                continue;

            const size_t idx = i * m_width + j;
            const double neighborhood = exp(-d2 / (2 * squared(radius * stdCoeff)));
            const double a = alpha * neighborhood;
            const double b = 1. - a; // beware, if b==0 then calculus goes wrong

            // calculate and store {w_{t+1}}^2
            squared_sum[idx] = squared(b) * squared_sum[idx] +
                               2 * a * b * w_coeff[idx] * wvprod[idx] +
                               squared(a) * data._sqsum[n];

            w_coeff[idx] *= b;  // do that before using in expression below

            partial_update(vsp, vind, vsz, codebook + idx * m_dim, a / w_coeff[idx]);

            if (w_coeff[idx] < EPSILON)
            {
                stabilize(idx);
            }
        }
    }
}

vector<bmu> Som::getBmus(const CSR& data) const
{
/*
    clock_t t1 = clock();
*/
    vector<bmu> bmus = vector<bmu>(data.nrows, bmu(0, DBL_MAX));

    for (size_t k=0; k < m_height*m_width; ++k)
    {
        const double * w = &codebook[k * m_dim];

#pragma omp parallel for

        for (idx_t i=0; i < (idx_t) data.nrows; ++i)
        {
            const int ind = data.indptr[i];
            const int vsz = data.indptr[i+1] - ind;
            const double dst = euclideanDistanceSq(&data.data[ind], &data.indices[ind], vsz, data._sqsum[i], w, squared_sum[k], w_coeff[k]);

            if (dst < bmus[i].dst)
            {
                bmus[i].idx = k;
                bmus[i].dst = dst;
            }
        }
    }
/*
    clock_t t2 = clock();

    if (m_verbose)
    {
        cout << " get bmus - elapsed " << (double)(t2-t1) / CLOCKS_PER_SEC << "s" << endl;
    }
*/
    return bmus;
}

/*
static double getQE(vector<bmu>& bmus)
{
    double qError = 0.;
    for (const bmu& star : bmus)
    {
        qError += sqrt(star.dst);
    }
    return qError / bmus.size();
}
*/

void Som::train(const CSR& data, size_t tmax,
           double radius0, double alpha0, double radiusN, double alphaN,
           double stdCoeff, cooling rcool, cooling acool)
{
    NBSTABLES = 0;

    double tm = 0.;

    if (m_verbose > 0)
    {
        cout << "Start learning..." << endl;

        tm = get_wall_time();
    }

    // allocate helpers internal arrays
    squared_sum = new double[m_height * m_width];
    w_coeff = new double[m_height * m_width];
    wvprod = new double[m_height * m_width];

#pragma omp parallel for
    // Init w^2 before training
    for (idx_t k=0; k < m_height * m_width; ++k)
    {
        squared_sum[k] = squared(codebook + k * m_dim, m_dim);
    }

    // init coefficients
    fill_n(w_coeff, m_height * m_width, 1.);

    double Qe = 0;
    const size_t percent = tmax / 100;

    default_random_engine rng(time(NULL));
    uniform_int_distribution<size_t> uidist(0, data.nrows-1);

    for (size_t t=1; t <= tmax; ++t)
    {
        double radius, alpha;
        const double ratio = ((double)t / tmax);

        switch (rcool) {
        case EXPONENTIAL:
            radius = radius0 * pow(radiusN / radius0, ratio);
            break;
        case LINEAR:
        default:
            radius = radiusN + (radius0 - radiusN) * (1. - ratio);
            break;
        }

        switch (acool) {
        case EXPONENTIAL:
            alpha = alpha0 * pow(alphaN / alpha0, ratio);
            break;
        case LINEAR:
        default:
            alpha = alphaN + (alpha0 - alphaN) * (1. - ratio);
            break;
        }

        const size_t k = uidist(rng);

        // Get the best match unit
        double dStar;
        const size_t kStar = getBmu(data, k, dStar);

        // Update network
        update(data, k, kStar, radius, alpha, stdCoeff);

        if (m_verbose > 1)
        {
            Qe += sqrt(dStar);  // approximated

            if (t % percent == 0)
            {
                cout << "  " << t / percent << "% (" << t << " / " << tmax << ")"
                    << " - r = " << radius << ", a = " << alpha
                    << " - approx. QE " << Qe / percent << endl;
                Qe = 0; // reinit Qe count
            }
        }
    }

    // stabilize all units
#pragma omp parallel for \
    schedule(static,1)

    for (idx_t k=0; k < m_height * m_width; ++k)
    {
        stabilize(k);
    }

    // dealloc internal helpers arrays
    delete [] squared_sum;
    delete [] w_coeff;
    delete [] wvprod;

    if (m_verbose > 0)
    {
        tm = get_wall_time() - tm;

        cout << "Finished: elapsed : " << tm << "s" << endl;
        if (m_verbose > 1)
        {
            cout << "  coeffs rescaled " << NBSTABLES << " times" << endl;
        }
/*
        // get all BMUs
        vector<bmu> bmus = getBmus(data);
        cout << "Quantization Error: " << getQE(bmus) << endl;
*/
    }
}


void Som::dump(const std::string& filename, bool binary) const
{
    ofstream myfile;
    myfile.open(filename, ios::binary);

    if (!myfile.is_open())
        throw "unable to open " + filename;

    myfile << m_height << ' ' << m_width << ' ' << m_dim << '\n';

    if (binary)
    {
        const size_t sz = m_height * m_width * m_dim;
        for (size_t k=0; k < sz; k += BUFSIZE)
        {
            size_t towrite = min<size_t>(BUFSIZE, sz - k);
            copy(codebook + k, codebook + k + towrite, iobuf);
            myfile.write((const char*)iobuf, towrite * sizeof(*iobuf));
        }
    }
    else
    {
        for (size_t k=0; k < m_height*m_width; ++k)
        {
            double *w = codebook + k * m_dim;
            for (size_t i=0; i < m_dim; ++i)
                myfile << w[i] << ' ';
            myfile << '\n';
        }
    }

    myfile.close();
}

/*
void Som::dumpSparse(const string& filename, double epsilon) const
{
    ofstream myfile;
    myfile.open(filename);

    if (!myfile.is_open())
        throw "unable to open " + filename;

    size_t cnt=0;

    //myfile << '#' << m_height << ' ' << m_width << ' ' << m_dim << endl;
    for (size_t k=0; k < m_height*m_width; ++k)
    {
        double *w = codebook + k * m_dim;
        for (size_t i=0; i<m_dim; ++i)
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

vector<label_counter> Som::calibrate(const dataset& dataSet) const
{
    vector<label_counter> mappings = vector<label_counter>(m_height*m_width);
    vector<bmu> bmus = getBmus(dataSet);
    for (int k=0; k < dataSet.nrows; ++k)
    {
        const bmu & star = bmus[k];
        mappings[star.idx][dataSet.labels[k]]++;
    }
    return mappings;
}

void Som::classify(const dataset& dataSet, const string& filename, bool cnt) const
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

#ifndef SPARSE_VEC_H
#define SPARSE_VEC_H

#include <cstdint>
#include <vector>
#include <cmath>

struct cell
{
    uint32_t idx;
    float val;
};

struct sparse_vec : private std::vector<cell>
{
    // store v^2
    float sumOfSquares;

    sparse_vec() :
        sumOfSquares(0.f)
    {}

    using std::vector<cell>::cbegin;
    using std::vector<cell>::cend;
    using std::vector<cell>::size;
    using std::vector<cell>::reserve;

    // only const version allowed
    inline const_reference back() const
    {
        return std::vector<cell>::back();
    }

    inline void push_back(const_reference c)
    {
        std::vector<cell>::push_back(c);
        sumOfSquares += c.val * c.val;
    }

    inline void clear()
    {
        std::vector<cell>::clear();
        sumOfSquares = 0.;
    }

    inline float norm() const
    {
        return std::sqrt(sumOfSquares);
    }

    void normalize();
};

#endif // SPARSE_VEC_H

#include "sparse_vec.h"


void sparse_vec::normalize()
{
    const float m = norm();
    sumOfSquares = 0.f; // reset sumOfSquares
    for (auto it=begin(); it!=end(); ++it)
    {
        it->val /= m;
        sumOfSquares += it->val * it->val;
    }
}

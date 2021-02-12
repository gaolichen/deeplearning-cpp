#pragma once

#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/data/monomorphic/generators/random.hpp>
#include <vector>
#include "../common.h"

namespace utf = boost::unit_test;
namespace bdata = boost::unit_test::data;
namespace tt = boost::test_tools;

const elem_t eps = 1.0e-8;
//#define inc 0.00001

#define MY_FLOAT_EQUAL(A, B, C) \
if (abs(A) < (C) || abs(B) < (C)) { \
    BOOST_TEST((A) - (B) == (float_type).0, tt::tolerance(C)); \
} else if (abs(A) < 1e-1 || abs(B) < 1e-1){ \
    BOOST_TEST((A) == (B), tt::tolerance(1000 * C)); \
} else { \
    BOOST_TEST((A) == (B), tt::tolerance(100 * C)); \
}

// simple test fixture class.
struct SimpleTestFixture
{
    SimpleTestFixture() {};
    ~SimpleTestFixture() {};
};


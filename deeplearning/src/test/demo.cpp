//Link to Boost
#define BOOST_TEST_DYN_LINK

//Define our Module name (prints at testing)
#define BOOST_TEST_MODULE BetheSolver UnitTests

//VERY IMPORTANT - include this last
#include <boost/test/included/unit_test.hpp>
//#include <boost/test/unit_test.hpp>

#include "test.h"

// test suite
BOOST_FIXTURE_TEST_SUITE(Demo_suite, SimpleTestFixture, * utf::label("UnityStartSystem"))

BOOST_AUTO_TEST_CASE(Demo)
{
    std::cout << "runnng demo." << std::endl;    
}

BOOST_AUTO_TEST_SUITE_END()

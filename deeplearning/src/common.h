#pragma once
#pragma warning(disable : 4267)
#define NATIVE_INT
#define NATIVE_FLOAT

#define PITEM(p) ((p) & 0xff)
#define PVAL(p) ((p) >> 8)
#define MAKEP(n, v) (((v) << 8) | (n))

#include <iostream>
#include <iterator>
#include <sstream>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <complex>
#include <string>
#include <sstream>
#include <ctime>
#include <limits>
#include <Eigen/Dense>
#include "Colormod.h"

#ifndef NATIVE_FLOAT
#include <boost/multiprecision/cpp_dec_float.hpp>
using boost::multiprecision::cpp_dec_float_50;
#endif

//#ifndef NATIVE_INT
#include <boost/multiprecision/cpp_int.hpp>
//#endif


#if WIN32
typedef __int64 i64;
typedef unsigned __int64 u64;
#else
typedef long long i64;
typedef unsigned long long u64;
#endif

typedef boost::multiprecision::uint128_t u128;

#ifdef NATIVE_INT
typedef u64 intN;
#define MAX_BIT 6
#else
#define MAX_BIT 7
//typedef boost::multiprecision::uint512_t intN;
typedef u128 intN;
#endif 

#ifdef NATIVE_FLOAT
typedef long double elem_t;
#else
typedef cpp_dec_float_50 elem_t; 
#endif

#define MAX_N (1<<MAX_BIT)

// MAX_COUNT is used by PartitionNumber and FreDistanceCalc classes.
// 1500 is the max value for PartitionNumber return values in range of type u128.
#define MAX_COUNT 1500
#define EPS 1e-10

#if MAX_COUNT > 400
#define PARTITION_NUMBER_128
typedef u128 count_t;
#else
typedef i64 count_t;
#endif

typedef std::complex<elem_t> var_t;
typedef Eigen::Matrix<var_t, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<var_t, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorI;
enum LogLevel {Info = 0, Warning = 1, Error = 2, Off = 100 };

extern double Pi;
extern long double logint[MAX_COUNT + 1];
extern long double logfactorial[MAX_COUNT + 1];
extern intN bionomial[MAX_N + 1][MAX_N + 1];

extern boost::random::mt19937 rng;
extern boost::random::uniform_01<elem_t> dist01;
extern std::ostream* logOut;
extern LogLevel currentLogLevel;

extern Color::Modifier red;
extern Color::Modifier def;
extern Color::Modifier green;
extern Color::Modifier yellow;

#define LOG(message, level) \
if (logOut && level >= currentLogLevel) { \
    if (level == Error) { \
        (*logOut) << red; \
    } else if (level == Warning) { \
        (*logOut) << yellow; \
    } else if (level == Info) {\
        (*logOut) << green; \
    } \
    (*logOut) << level << '\t' << message << def << std::endl; \
}

#define SET_LOG_LEVEL(level) \
currentLogLevel = level; \
if (level != Off) { \
    logOut = &std::cout; \
} else { \
    logOut = NULL; \
}

long double LogInt(int n);

long double LogFactorial(int n);

// return random integer between 0 and max, including max
i64 random64(i64 max);

// return random integer between 0 and max, including max
u128 random128(u128 max);

elem_t random(elem_t min, elem_t max);

var_t randomComplex(elem_t minR, elem_t maxR);

double EuclideanDist(double x1, double y1, double x2, double y2);
double EuclideanDistSquare(double x1, double y1, double x2, double y2);

std::ostream& operator<< (std::ostream& out, LogLevel level);

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v)
{
  out << '[';
  if (!v.empty()) {
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b";
  }
  //out << "\b\b]";
  out << ']';
  return out;
}

template<class T> std::string ToString(T a)
{
	std::ostringstream oss;
	oss << a;
	return oss.str();
};

template<class T> T ToExpression(std::string str, T defaultValue)
{
    T a;
	std::istringstream iss(str);
	if (iss >> a) {
	    return a;
	} else {
	    return defaultValue;
	}
};

template <typename T>
T Sum(const std::vector<T>& v)
{
    T ret = (T)0;
    for (int i = 0; i < v.size(); i++) ret += v[i];
    return ret;
}

template <typename T> 
void PrintVector(std::vector<T>& v)
{
	std::cout << '[';
	for (int i = 0; i < v.size(); i++) {
		if (i > 0) std::cout << ' ';
		std::cout << v[i];
	}

	std::cout << ']';
}

template <typename T> 
std::string VectorToString(std::vector<T>& v)
{
    std::string ret = "[";
	for (int i = 0; i < v.size(); i++) {
		if (i > 0) ret += " ";
		ret += ToString(v[i]);
	}

	return ret + "]";
}

template <typename T> 
int CompareVector(const std::vector<T>& v1, const std::vector<T>& v2)
{
	for (int i = 0; i < v1.size() && i < v2.size(); i++) {
	    if (v1[i] < v2[i]) return -1;
	    else if (v1[i] > v2[i]) return 1;
	}
	
	return v1.size() - v2.size();
}

template <typename T>
T arrayDot(T *a1, T *a2, int size)
{
    T ret = (T)0;
    for (int i = 0; i < size; i++) ret += a1[i] * a2[i];
    return ret;
}

intN BioCoeff(int n, int k);

bool isSingular(var_t u, elem_t eps = EPS);

var_t e2ip(var_t u, elem_t eps = EPS);

var_t e2ip(const Vector &u, elem_t eps = EPS);

elem_t momentum(Vector u, elem_t eps = EPS);

var_t sMatrix(var_t u1, var_t u2);

elem_t floatMod(elem_t num1, elem_t num2);

bool isZero(const var_t &val);

var_t chop(var_t v, elem_t eps = EPS);

void chop(Vector &v, elem_t eps = EPS);

class Stopwatch
{
private:
	clock_t start;
public:
	Stopwatch();
	void Restart();
	double Elapsed(bool restart = false);
    std::string Now();
    std::string Log(bool restart = false);
};

#pragma once
#pragma warning(disable : 4267)
#define NATIVE_INT
#define NATIVE_FLOAT
//#define EIGEN_USE_BLAS	

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
#include <chrono>
#include <Eigen/Dense>
#include "Colormod.h"

#if WIN32
typedef __int64 i64;
typedef unsigned __int64 u64;
#else
typedef long long i64;
typedef unsigned long long u64;
#endif

typedef u64 intN;
typedef double data_t;

#define EPS 1e-10

typedef Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> MatrixI;
typedef Eigen::Matrix<data_t, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<data_t, 1, Eigen::Dynamic> RVector;
typedef Eigen::Array<data_t, Eigen::Dynamic, Eigen::Dynamic> Array;
typedef Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> PermutationMatrix;
enum LogLevel {Info = 0, Warning = 1, Error = 2, Off = 100 };

extern double Pi;

extern boost::random::mt19937 rng;
extern boost::random::uniform_01<data_t> dist01;
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

data_t random(data_t min, data_t max);

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

std::vector<int> pickRandomIndexInner(int range, int n);

std::vector<int> pickRandomIndex(int range, int n);

class Stopwatch
{
private:
	std::chrono::steady_clock::time_point start;
public:
	Stopwatch();
	void Restart();
	double Elapsed(bool restart = false);
    std::string Now();
    std::string Log(bool restart = false);
};

class DPLException : public std::exception
{
private:
	std::string _message;
public:
	DPLException(std::string message) {
		this->_message = message;
	}

	virtual const char* what() const noexcept {
		return _message.c_str();
	}
};


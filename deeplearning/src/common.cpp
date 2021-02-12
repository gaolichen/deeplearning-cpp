#include "common.h"
#include <limits>
#include <cmath>
#include <ctime>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>

boost::random::mt19937 rng(time(0));
boost::random::uniform_01<elem_t> dist01;
std::ostream* logOut = &std::cout;
LogLevel currentLogLevel = Error;

Color::Modifier red(Color::FG_RED);
Color::Modifier def(Color::FG_DEFAULT);
Color::Modifier green(Color::FG_GREEN);
Color::Modifier yellow(Color::FG_YELLOW);


long double logint[MAX_COUNT + 1];
long double logfactorial[MAX_COUNT + 1];
double Pi = acos(-1.0);
intN bionomial[MAX_N + 1][MAX_N + 1];

std::ostream& operator<< (std::ostream& out, LogLevel level) {
    if (level == Info) {
        out << "Info:";
    } else if (level == Warning) {
        out << "Warning:";
    } else if (level == Error) {
        out << "Error:";
    } else {
        // do nothing.
    }
}

long double LogInt(int n)
{
	if (n == 0) {
		std::cout << "error! log(0)" << std::endl;
		return std::numeric_limits<long double>::max();
	}
	if (n == 1) return 0.0;

	if (logint[n] == 0.0) {
		logint[n] = logl(n);
	}
	
	return logint[n];
}

long double LogFactorial(int n)
{
	if (n == 1 || n == 0) return 0.0;

	if (logfactorial[n] == 0.0) {
		for (int i = 2; i <= n; i++) {
			logfactorial[n] += LogInt(i);
		}
	}

	return logfactorial[n];
}

elem_t random(elem_t min, elem_t max)
{
    return min + (max - min) * dist01(rng);
}

i64 random64(i64 max)
{
    boost::random::uniform_int_distribution<i64> dist(0, max);
    return dist(rng);
}

u128 random128(u128 max)
{
    boost::random::uniform_int_distribution<u64> high(0, (u64)(max >> 64));
    boost::random::uniform_int_distribution<u64> low(0, (u64)(max & std::numeric_limits<u64>::max()));

    return (((u128)high(rng)) << 64) | low(rng);
}

var_t randomComplex(elem_t minR, elem_t maxR) {
    elem_t r;
    while(true) {
        r = random(minR, maxR);
        if (random(0.0, maxR * maxR - minR * minR) < (r * r - minR * minR)) {
            break;
        }
    }
    elem_t phi = random(0.0, 2 * Pi);
    return var_t(r * cos(phi), r * sin(phi));
}



double EuclideanDist(double x1, double y1, double x2, double y2)
{
	return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

double EuclideanDistSquare(double x1, double y1, double x2, double y2)
{
	return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}

intN BioCoeff(int n, int k)
{
	if (k > n) return 0;
	if (k == 0 || k == n) return 1;
	intN& ret = bionomial[n][k];
	if (ret == 0) {
		ret = BioCoeff(n - 1, k) + BioCoeff(n - 1, k - 1);
	}

	return ret;
}

bool isSingular(var_t u, elem_t eps) {
    return std::abs(u.real()) < eps && (std::abs(u.imag() + 0.5) < eps || std::abs(u.imag() - 0.5) < eps);
}

var_t e2ip(var_t u, elem_t eps) {
    if (isSingular(u)) {
        if (u.imag() > (elem_t)0.0) {
            return (elem_t)1.0;
        } else {
            return (elem_t)-1.0;
        }
    }
    return (u + var_t(0.0, 0.5))/(u - var_t(0.0, 0.5));
/*    if (abs(u.real()) > EPS) {
        return (u + var_t(0.0, 0.5))/(u - var_t(0.0, 0.5));
    } else {
        if (abs(u.imag() + 0.5) <= EPS) {
            return var_t(0, -EPS);
        } else if (abs(u.imag() - 0.5) <= EPS) {
            return var_t(0, 1/EPS);
        } else {
            return (u + var_t(0.0, 0.5))/(u - var_t(0.0, 0.5));
        }
    }*/
}

var_t e2ip(const Vector &u, elem_t eps) {
    var_t ret((elem_t)1.0);
    for (int i = 0; i < u.size(); i++) {
        ret *= e2ip(u[i], eps);
    }
    return ret;
}

elem_t momentum(Vector u, elem_t eps) {
    var_t ret = e2ip(u, eps);
//    return log(ret) * var_t(0.0, -1.0);
    return std::arg(ret);
}

var_t sMatrix(var_t u1, var_t u2) {
    return (u1 - u2 - var_t(0, 1.0))/(u1 - u2 + var_t(0, 1.0));
}

elem_t floatMod(elem_t num1, elem_t num2) {
    return num1 - floor((num1 + num2 * 0.5)/num2) * num2;
}

bool isZero(const var_t &val) {
    return abs(val.real()) < EPS && abs(val.imag()) < EPS;
}

var_t chop(var_t v, elem_t eps) {
    elem_t re = v.real();
    elem_t im = v.imag();
    if (std::abs(re) < eps) re = 0.0L;
    if (std::abs(im) < eps) im = 0.0L;
    return var_t(re, im);
}

void chop(Vector &v, elem_t eps) {
    for (int i = 0; i < v.size(); i++) {
        v[i] = chop(v[i], eps);
    }
}


Stopwatch::Stopwatch()
{
    start = clock();
}

void Stopwatch::Restart()
{
        start = clock();
}

std::string Stopwatch::Log(bool restart)
{
    double ret = (clock() - start) / (double)CLOCKS_PER_SEC;
    std::string text = ToString(ret) + " seconds elapsed, current time " + Now();
    
    if (restart) {
        Restart();
    }
    
    return text;
}

std::string Stopwatch::Now()
{
    time_t t = time(0);   // get time now
    struct tm * now = localtime(&t);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", now);
    return buf;
}

/// return how much time in second since last Start() function call
double Stopwatch::Elapsed(bool restart)
{
        double ret = (clock() - start) / (double)CLOCKS_PER_SEC;
        if (restart) Restart();
        return ret;
}

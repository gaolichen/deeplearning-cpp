#include "common.h"
#include <limits>
#include <cmath>
#include <ctime>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>

boost::random::mt19937 rng(time(0));
boost::random::uniform_01<data_t> dist01;
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

data_t random(data_t min, data_t max)
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

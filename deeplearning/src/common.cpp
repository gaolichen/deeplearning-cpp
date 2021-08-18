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

double Pi = acos(-1.0);

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

data_t random(data_t min, data_t max)
{
    return min + (max - min) * dist01(rng);
}

i64 random64(i64 max)
{
    boost::random::uniform_int_distribution<i64> dist(0, max);
    return dist(rng);
}

double EuclideanDist(double x1, double y1, double x2, double y2)
{
	return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

double EuclideanDistSquare(double x1, double y1, double x2, double y2)
{
	return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}

std::vector<int> pickRandomIndexInner(int range, int n) {
    std::vector<int> ret(n);
    int beginIndex = 0;
    while (beginIndex < n) {
        for (int i = beginIndex; i < n; i++) {
            ret[i] = random64(range - 1);
        }
        std::sort(ret.begin(), ret.end());
        beginIndex = std::unique(ret.begin(), ret.end()) - ret.begin();
    }
    return ret;
}

std::vector<int> pickRandomIndex(int range, int n) {
    if (n + n < range) {
        return pickRandomIndexInner(range, n);
    } else {
        std::vector<int> v = pickRandomIndexInner(range, range - n);
        std::vector<int> ret(n);
        int j = 0, k = 0;
        for (int i = 0; i < range; i++) {
            if (j < v.size() && i == v[j]) {
                j++;
                continue;
            } else {
                ret[k++] = i;
            }
        }
        
        return ret;
    }
}

Stopwatch::Stopwatch()
{
    start = std::chrono::steady_clock::now();
}

void Stopwatch::Restart()
{
        start = std::chrono::steady_clock::now();;
}

std::string Stopwatch::Log(bool restart)
{
    double ret = Elapsed(restart);
    return ToString(ret) + " seconds elapsed, current time " + Now();
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
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    int cnt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double ret = cnt / 1000.0;
    if (restart) Restart();
    return ret;
}

#pragma once
#include "common.h"

#define SHOW_PROGRESS(bar, step, message) \
    (bar).update((step), "", false); \
    std::cout << message; \
    if (step == bar.getTotSteps()) { \
        std::cout << std::endl; \
    } else { \
        std::cout << '\r'; \
    } \
    std::cout.flush();

#define PROGRESS_DONE(bar, message) SHOW_PROGRESS(bar, bar.getTotSteps(), message)

class ProgressBar
{
private:
    std::string _header;
    int _totSteps;
    int _barWidth;
public:
    ProgressBar(std::string header, int totSteps, int barWidth = 50);
    
    void update(int step, std::string message = "", bool print_end = true);
    
    int getTotSteps() {
        return this->_totSteps;
    }
};

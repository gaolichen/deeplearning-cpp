#include "progressbar.h"

ProgressBar::ProgressBar(std::string header, int totSteps, int barWidth) {
    _header = header;
    _totSteps = totSteps;
    _barWidth = barWidth;
}

void ProgressBar::update(int step, std::string message, bool print_end) {
    double percentage = step / (double)_totSteps;
    int len = (int)floor(percentage * _barWidth + 0.5);
    std::cout << _header << '[';
    for (int i = 0; i < _barWidth; i++) {
        if (i < len) {
            std::cout << '=';
        } else if (i == len) {
            std::cout << '>';
        } else {
            std::cout << '.';
        }
    }
    std::cout << "] " << (int)floor(percentage * 100 + 0.5) << "% " << message;
    if (!print_end) {
        return;
    }
    
    if (step == _totSteps) {
        std::cout << std::endl;
    } else {
        std::cout << "\r";
    }
    
    std::cout.flush();
}

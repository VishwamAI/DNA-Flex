#pragma once

#include <string>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dnaflex {
namespace parsers {

struct MSAProfileHandler {
    std::vector<std::string> sequences;
    std::vector<std::vector<double>> weights;
    std::vector<double> conservation;
    
    void add_sequence(const std::string& seq);
    void compute_profile();
    py::array_t<double> get_position_weights() const;
    py::array_t<double> get_conservation() const;
}; 

} // namespace parsers
} // namespace dnaflex
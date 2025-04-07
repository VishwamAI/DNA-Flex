#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace dnaflex {
namespace parsers {

class MSAProfile {
public:
    MSAProfile();
    ~MSAProfile();

    void add_sequence(const std::string& sequence);
    void compute_profile();
    py::array_t<double> get_position_weights() const;
    py::array_t<double> get_conservation_scores() const;
    
    double get_sequence_identity() const;
    std::vector<double> get_gap_frequencies() const;
    std::map<char, std::vector<double>> get_nucleotide_frequencies() const;
    std::string get_consensus_sequence() const;
    std::vector<std::string> get_variable_regions() const;
    size_t get_alignment_length() const;
    size_t get_sequence_count() const;
    double get_average_sequence_length() const;

private:
    struct ProfileImpl;
    std::unique_ptr<ProfileImpl> impl_;
    
    void validate_sequence(const std::string& sequence) const;
    void update_position_weights();
    void calculate_conservation();
};

} // namespace parsers
} // namespace dnaflex
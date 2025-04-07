#include "msa_profile_pybind.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace dnaflex {
namespace msa {

struct MSAProfile::ProfileImpl {
    std::vector<std::string> sequences;
    std::vector<std::vector<double>> position_weights;
    std::vector<double> conservation_scores;
    size_t alignment_length = 0;
    static const std::vector<char> valid_nucleotides;
};

const std::vector<char> MSAProfile::ProfileImpl::valid_nucleotides = {'A', 'T', 'G', 'C', '-'};

MSAProfile::MSAProfile() : impl_(std::make_unique<ProfileImpl>()) {}
MSAProfile::~MSAProfile() = default;

void MSAProfile::add_sequence(const std::string& sequence) {
    validate_sequence(sequence);
    
    if (impl_->sequences.empty()) {
        impl_->alignment_length = sequence.length();
    } else if (sequence.length() != impl_->alignment_length) {
        throw std::invalid_argument("All sequences must have the same length");
    }
    
    impl_->sequences.push_back(sequence);
}

void MSAProfile::validate_sequence(const std::string& sequence) const {
    for (char c : sequence) {
        char upper_c = std::toupper(c);
        auto it = std::find_if(impl_->valid_nucleotides.begin(),
                              impl_->valid_nucleotides.end(),
                              [upper_c](const char& n) { return n == upper_c; });
        if (it == impl_->valid_nucleotides.end()) {
            throw std::invalid_argument("Invalid nucleotide in sequence: " + std::string(1, c));
        }
    }
}

void MSAProfile::compute_profile() {
    if (impl_->sequences.empty()) {
        throw std::runtime_error("No sequences added to profile");
    }
    
    update_position_weights();
    calculate_conservation();
}

void MSAProfile::update_position_weights() {
    impl_->position_weights.clear();
    impl_->position_weights.resize(impl_->alignment_length);
    
    for (size_t pos = 0; pos < impl_->alignment_length; ++pos) {
        std::map<char, double> nucl_count;
        for (const auto& seq : impl_->sequences) {
            char base = std::toupper(seq[pos]);
            nucl_count[base]++;
        }
        
        std::vector<double> pos_weights;
        for (char nucl : impl_->valid_nucleotides) {
            double freq = nucl_count[nucl] / impl_->sequences.size();
            pos_weights.push_back(freq);
        }
        impl_->position_weights[pos] = pos_weights;
    }
}

void MSAProfile::calculate_conservation() {
    impl_->conservation_scores.clear();
    impl_->conservation_scores.resize(impl_->alignment_length);
    
    for (size_t pos = 0; pos < impl_->alignment_length; ++pos) {
        double entropy = 0.0;
        for (const auto& weight : impl_->position_weights[pos]) {
            if (weight > 0) {
                entropy -= weight * std::log2(weight);
            }
        }
        // Normalize by max possible entropy (log2 of number of possible states)
        impl_->conservation_scores[pos] = 1.0 - (entropy / std::log2(5.0));
    }
}

py::array_t<double> MSAProfile::get_position_weights() const {
    if (impl_->position_weights.empty()) {
        throw std::runtime_error("Profile not computed. Call compute_profile() first.");
    }
    
    std::vector<double> flattened;
    flattened.reserve(impl_->alignment_length * 5);
    for (const auto& pos : impl_->position_weights) {
        flattened.insert(flattened.end(), pos.begin(), pos.end());
    }
    
    std::vector<ssize_t> shape = {static_cast<ssize_t>(impl_->alignment_length), 5};
    std::vector<ssize_t> strides = {static_cast<ssize_t>(5 * sizeof(double)), sizeof(double)};
    
    auto* data = new double[flattened.size()];
    std::copy(flattened.begin(), flattened.end(), data);
    
    return py::array_t<double>(
        shape,
        strides,
        data,
        py::capsule(data, [](void *f) {
            delete[] reinterpret_cast<double*>(f);
        })
    );
}

py::array_t<double> MSAProfile::get_conservation_scores() const {
    if (impl_->conservation_scores.empty()) {
        throw std::runtime_error("Profile not computed. Call compute_profile() first.");
    }
    
    auto* data = new double[impl_->conservation_scores.size()];
    std::copy(impl_->conservation_scores.begin(), impl_->conservation_scores.end(), data);
    
    return py::array_t<double>(
        {static_cast<ssize_t>(impl_->conservation_scores.size())},
        {sizeof(double)},
        data,
        py::capsule(data, [](void *f) {
            delete[] reinterpret_cast<double*>(f);
        })
    );
}

double MSAProfile::get_sequence_identity() const {
    if (impl_->sequences.size() < 2) return 1.0;
    
    double total_identity = 0.0;
    int comparisons = 0;
    
    for (size_t i = 0; i < impl_->sequences.size(); ++i) {
        for (size_t j = i + 1; j < impl_->sequences.size(); ++j) {
            int matches = 0;
            for (size_t pos = 0; pos < impl_->alignment_length; ++pos) {
                if (std::toupper(impl_->sequences[i][pos]) == 
                    std::toupper(impl_->sequences[j][pos])) {
                    matches++;
                }
            }
            total_identity += static_cast<double>(matches) / impl_->alignment_length;
            comparisons++;
        }
    }
    
    return total_identity / comparisons;
}

std::vector<double> MSAProfile::get_gap_frequencies() const {
    std::vector<double> gap_freqs(impl_->alignment_length, 0.0);
    
    for (size_t pos = 0; pos < impl_->alignment_length; ++pos) {
        int gap_count = 0;
        for (const auto& seq : impl_->sequences) {
            if (seq[pos] == '-') {
                gap_count++;
            }
        }
        gap_freqs[pos] = static_cast<double>(gap_count) / impl_->sequences.size();
    }
    
    return gap_freqs;
}

std::map<char, std::vector<double>> MSAProfile::get_nucleotide_frequencies() const {
    std::map<char, std::vector<double>> freqs;
    for (char nucl : impl_->valid_nucleotides) {
        freqs[nucl] = std::vector<double>(impl_->alignment_length, 0.0);
    }
    
    for (size_t pos = 0; pos < impl_->alignment_length; ++pos) {
        std::map<char, int> counts;
        for (const auto& seq : impl_->sequences) {
            char base = std::toupper(seq[pos]);
            counts[base]++;
        }
        
        for (const auto& [nucl, count] : counts) {
            freqs[nucl][pos] = static_cast<double>(count) / impl_->sequences.size();
        }
    }
    
    return freqs;
}

std::string MSAProfile::get_consensus_sequence() const {
    std::string consensus;
    consensus.reserve(impl_->alignment_length);
    
    for (size_t pos = 0; pos < impl_->alignment_length; ++pos) {
        std::map<char, int> counts;
        for (const auto& seq : impl_->sequences) {
            counts[std::toupper(seq[pos])]++;
        }
        
        auto max_it = std::max_element(
            counts.begin(), counts.end(),
            [](const std::pair<const char, int>& p1, const std::pair<const char, int>& p2) {
                return p1.second < p2.second;
            }
        );
        consensus += max_it->first;
    }
    
    return consensus;
}

std::vector<std::string> MSAProfile::get_variable_regions() const {
    std::vector<std::string> regions;
    const double CONSERVATION_THRESHOLD = 0.7;
    
    if (impl_->conservation_scores.empty()) {
        throw std::runtime_error("Profile not computed. Call compute_profile() first.");
    }
    
    std::string consensus = get_consensus_sequence();
    size_t start = 0;
    bool in_variable_region = false;
    
    for (size_t i = 0; i < impl_->alignment_length; ++i) {
        if (impl_->conservation_scores[i] < CONSERVATION_THRESHOLD) {
            if (!in_variable_region) {
                start = i;
                in_variable_region = true;
            }
        } else if (in_variable_region) {
            regions.push_back(consensus.substr(start, i - start));
            in_variable_region = false;
        }
    }
    
    if (in_variable_region) {
        regions.push_back(consensus.substr(start));
    }
    
    return regions;
}

size_t MSAProfile::get_alignment_length() const {
    return impl_->alignment_length;
}

size_t MSAProfile::get_sequence_count() const {
    return impl_->sequences.size();
}

double MSAProfile::get_average_sequence_length() const {
    if (impl_->sequences.empty()) return 0.0;
    
    double total_length = 0.0;
    for (const auto& seq : impl_->sequences) {
        total_length += static_cast<double>(
            std::count_if(seq.begin(), seq.end(),
                         [](const char& c) { return c != '-'; }));
    }
    
    return total_length / impl_->sequences.size();
}

} // namespace msa
} // namespace dnaflex

PYBIND11_MODULE(msa_profile, m) {
    m.doc() = "Multiple Sequence Alignment Profile module for DNA sequences";
    
    py::class_<dnaflex::msa::MSAProfile>(m, "MSAProfile")
        .def(py::init<>())
        .def("add_sequence", &dnaflex::msa::MSAProfile::add_sequence)
        .def("compute_profile", &dnaflex::msa::MSAProfile::compute_profile)
        .def("get_position_weights", &dnaflex::msa::MSAProfile::get_position_weights)
        .def("get_conservation_scores", &dnaflex::msa::MSAProfile::get_conservation_scores)
        .def("get_sequence_identity", &dnaflex::msa::MSAProfile::get_sequence_identity)
        .def("get_gap_frequencies", &dnaflex::msa::MSAProfile::get_gap_frequencies)
        .def("get_nucleotide_frequencies", &dnaflex::msa::MSAProfile::get_nucleotide_frequencies)
        .def("get_consensus_sequence", &dnaflex::msa::MSAProfile::get_consensus_sequence)
        .def("get_variable_regions", &dnaflex::msa::MSAProfile::get_variable_regions)
        .def("get_alignment_length", &dnaflex::msa::MSAProfile::get_alignment_length)
        .def("get_sequence_count", &dnaflex::msa::MSAProfile::get_sequence_count)
        .def("get_average_sequence_length", &dnaflex::msa::MSAProfile::get_average_sequence_length);
}
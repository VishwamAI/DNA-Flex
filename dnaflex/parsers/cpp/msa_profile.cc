#include "msa_profile_lib.h"
#include <algorithm>
#include <numeric>
#include <map>
#include <cmath>

namespace dnaflex {
namespace parsers {

struct MSAProfile::ProfileImpl {
    std::vector<std::string> sequences;
    std::vector<std::vector<double>> position_weights;
    std::vector<double> conservation_scores;
    std::map<char, std::vector<double>> nucleotide_freqs;
    double avg_identity;
};

MSAProfile::MSAProfile() : impl_(std::make_unique<ProfileImpl>()) {}
MSAProfile::~MSAProfile() = default;

void MSAProfile::add_sequence(const std::string& sequence) {
    validate_sequence(sequence);
    impl_->sequences.push_back(sequence);
}

void MSAProfile::compute_profile() {
    if (impl_->sequences.empty()) {
        return;
    }

    const size_t seq_len = impl_->sequences[0].length();
    impl_->position_weights.resize(seq_len);
    impl_->conservation_scores.resize(seq_len);
    
    // Initialize nucleotide frequency maps
    for (char base : {'A', 'T', 'G', 'C', '-'}) {
        impl_->nucleotide_freqs[base] = std::vector<double>(seq_len, 0.0);
    }

    // Calculate nucleotide frequencies at each position
    for (size_t pos = 0; pos < seq_len; ++pos) {
        std::map<char, int> counts;
        for (const auto& seq : impl_->sequences) {
            char base = std::toupper(seq[pos]);
            counts[base]++;
            impl_->nucleotide_freqs[base][pos] = counts[base] / static_cast<double>(impl_->sequences.size());
        }
        
        // Calculate position conservation using Shannon entropy
        double entropy = 0.0;
        for (const auto& [base, freq] : impl_->nucleotide_freqs) {
            if (freq[pos] > 0) {
                entropy -= freq[pos] * std::log2(freq[pos]);
            }
        }
        impl_->conservation_scores[pos] = 1.0 - (entropy / std::log2(5.0)); // 5 possible states (ATGC-)
    }

    // Calculate average sequence identity
    impl_->avg_identity = 0.0;
    int comparisons = 0;
    for (size_t i = 0; i < impl_->sequences.size(); ++i) {
        for (size_t j = i + 1; j < impl_->sequences.size(); ++j) {
            int matches = 0;
            for (size_t pos = 0; pos < seq_len; ++pos) {
                if (impl_->sequences[i][pos] == impl_->sequences[j][pos]) {
                    matches++;
                }
            }
            impl_->avg_identity += static_cast<double>(matches) / seq_len;
            comparisons++;
        }
    }
    if (comparisons > 0) {
        impl_->avg_identity /= comparisons;
    }
}

py::array_t<double> MSAProfile::get_position_weights() const {
    if (impl_->sequences.empty()) {
        return py::array_t<double>();
    }
    
    auto result = py::array_t<double>(impl_->position_weights.size());
    auto buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < impl_->position_weights.size(); ++i) {
        buf(i) = impl_->conservation_scores[i];
    }
    return result;
}

py::array_t<double> MSAProfile::get_conservation_scores() const {
    if (impl_->sequences.empty()) {
        return py::array_t<double>();
    }
    
    auto result = py::array_t<double>(impl_->conservation_scores.size());
    auto buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < impl_->conservation_scores.size(); ++i) {
        buf(i) = impl_->conservation_scores[i];
    }
    return result;
}

double MSAProfile::get_sequence_identity() const {
    return impl_->avg_identity;
}

std::vector<double> MSAProfile::get_gap_frequencies() const {
    if (impl_->sequences.empty()) {
        return std::vector<double>();
    }
    return impl_->nucleotide_freqs.at('-');
}

std::map<char, std::vector<double>> MSAProfile::get_nucleotide_frequencies() const {
    return impl_->nucleotide_freqs;
}

std::string MSAProfile::get_consensus_sequence() const {
    if (impl_->sequences.empty()) {
        return "";
    }
    
    std::string consensus;
    const size_t seq_len = impl_->sequences[0].length();
    
    for (size_t pos = 0; pos < seq_len; ++pos) {
        char max_base = 'N';
        double max_freq = 0.0;
        
        for (const auto& [base, freqs] : impl_->nucleotide_freqs) {
            if (base != '-' && freqs[pos] > max_freq) {
                max_freq = freqs[pos];
                max_base = base;
            }
        }
        consensus += max_base;
    }
    
    return consensus;
}

std::vector<std::string> MSAProfile::get_variable_regions() const {
    if (impl_->sequences.empty()) {
        return std::vector<std::string>();
    }
    
    std::vector<std::string> regions;
    const double VARIABILITY_THRESHOLD = 0.5;
    const size_t MIN_REGION_LENGTH = 3;
    
    size_t start = 0;
    bool in_variable_region = false;
    
    for (size_t i = 0; i < impl_->conservation_scores.size(); ++i) {
        if (impl_->conservation_scores[i] < VARIABILITY_THRESHOLD) {
            if (!in_variable_region) {
                start = i;
                in_variable_region = true;
            }
        } else if (in_variable_region) {
            if (i - start >= MIN_REGION_LENGTH) {
                regions.push_back(impl_->sequences[0].substr(start, i - start));
            }
            in_variable_region = false;
        }
    }
    
    // Check last region
    if (in_variable_region && impl_->conservation_scores.size() - start >= MIN_REGION_LENGTH) {
        regions.push_back(impl_->sequences[0].substr(start));
    }
    
    return regions;
}

size_t MSAProfile::get_alignment_length() const {
    return impl_->sequences.empty() ? 0 : impl_->sequences[0].length();
}

size_t MSAProfile::get_sequence_count() const {
    return impl_->sequences.size();
}

double MSAProfile::get_average_sequence_length() const {
    if (impl_->sequences.empty()) {
        return 0.0;
    }
    
    double total_length = 0.0;
    for (const auto& seq : impl_->sequences) {
        total_length += std::count_if(seq.begin(), seq.end(), 
            [](char c) { return c != '-'; });
    }
    return total_length / impl_->sequences.size();
}

void MSAProfile::validate_sequence(const std::string& sequence) const {
    if (!impl_->sequences.empty() && sequence.length() != impl_->sequences[0].length()) {
        throw std::runtime_error("All sequences must have the same length");
    }
    
    for (char c : sequence) {
        if (c != 'A' && c != 'T' && c != 'G' && c != 'C' && 
            c != 'a' && c != 't' && c != 'g' && c != 'c' && c != '-') {
            throw std::runtime_error("Invalid nucleotide in sequence");
        }
    }
}

} // namespace parsers
} // namespace dnaflex
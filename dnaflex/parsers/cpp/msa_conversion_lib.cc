#include "msa_conversion_lib.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace dnaflex {
namespace parsers {

MSAConverter::MSAConverter() {}
MSAConverter::~MSAConverter() {}

std::vector<std::vector<char>> MSAConverter::fasta_to_matrix(const std::string& filename) {
    FastaIterator iterator(filename);
    std::vector<FastaEntry> entries;
    FastaEntry entry;
    
    while (iterator.next(entry)) {
        entries.push_back(entry);
    }

    if (entries.empty()) {
        throw std::runtime_error("No sequences found in FASTA file");
    }

    if (!check_sequence_lengths(entries)) {
        throw std::runtime_error("Sequences have different lengths");
    }

    std::vector<std::vector<char>> matrix(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        const std::string& seq = entries[i].sequence;
        matrix[i].reserve(seq.length());
        std::copy(seq.begin(), seq.end(), std::back_inserter(matrix[i]));
    }

    return matrix;
}

bool MSAConverter::matrix_to_fasta(const std::vector<std::vector<char>>& matrix,
                                 const std::vector<std::string>& headers,
                                 const std::string& output_file) {
    if (!validate_alignment(matrix)) {
        return false;
    }

    if (headers.size() != matrix.size()) {
        return false;
    }

    std::ofstream out(output_file);
    if (!out.is_open()) {
        return false;
    }

    for (size_t i = 0; i < matrix.size(); ++i) {
        out << ">" << headers[i] << "\n";
        const auto& sequence = matrix[i];
        for (size_t j = 0; j < sequence.size(); ++j) {
            out << sequence[j];
            if ((j + 1) % 60 == 0) {  // Standard FASTA line length
                out << "\n";
            }
        }
        if (sequence.size() % 60 != 0) {
            out << "\n";
        }
    }

    return true;
}

bool MSAConverter::validate_alignment(const std::vector<std::vector<char>>& matrix) {
    if (matrix.empty()) {
        return false;
    }

    size_t length = matrix[0].size();
    return std::all_of(matrix.begin(), matrix.end(),
                      [length](const auto& seq) { return seq.size() == length; });
}

size_t MSAConverter::get_alignment_length(const std::vector<std::vector<char>>& matrix) {
    return matrix.empty() ? 0 : matrix[0].size();
}

size_t MSAConverter::get_sequence_count(const std::vector<std::vector<char>>& matrix) {
    return matrix.size();
}

bool MSAConverter::check_sequence_lengths(const std::vector<FastaEntry>& entries) {
    if (entries.empty()) {
        return true;
    }
    
    size_t length = entries[0].sequence.length();
    return std::all_of(entries.begin(), entries.end(),
                      [length](const auto& entry) { return entry.sequence.length() == length; });
}

} // namespace parsers
} // namespace dnaflex
#include "msa_conversion_lib.h"
#include <algorithm>
#include <stdexcept>
#include <pybind11/numpy.h>

namespace dnaflex {
namespace parsers {

MSAConverter::MSAConverter() : max_seq_len_(0) {}
MSAConverter::~MSAConverter() = default;

py::array_t<char> MSAConverter::fasta_to_matrix(const std::string& filename) {
    fasta_iter_ = std::make_unique<FastaIterator>(filename);
    sequences_.clear();
    headers_.clear();
    max_seq_len_ = 0;

    // First pass: collect sequences and find max length
    FastaEntry entry;
    while (fasta_iter_->next(entry)) {
        sequences_.push_back(entry.sequence);
        headers_.push_back(entry.header);
        max_seq_len_ = std::max(max_seq_len_, entry.sequence.length());
    }

    if (sequences_.empty()) {
        throw std::runtime_error("No sequences found in FASTA file");
    }

    // Create numpy array for the alignment matrix
    auto result = py::array_t<char>({sequences_.size(), max_seq_len_});
    auto buf = result.mutable_unchecked<2>();

    // Fill the matrix
    for (size_t i = 0; i < sequences_.size(); ++i) {
        const auto& seq = sequences_[i];
        for (size_t j = 0; j < max_seq_len_; ++j) {
            buf(i, j) = j < seq.length() ? seq[j] : '-';
        }
    }

    return result;
}

bool MSAConverter::matrix_to_fasta(const py::array_t<char>& matrix,
                                 const std::vector<std::string>& headers,
                                 const std::string& output_file) {
    if (matrix.ndim() != 2) {
        throw std::runtime_error("Matrix must be 2-dimensional");
    }

    if (headers.size() != static_cast<size_t>(matrix.shape(0))) {
        throw std::runtime_error("Number of headers must match number of sequences");
    }

    std::ofstream out(output_file);
    if (!out) {
        return false;
    }

    auto buf = matrix.unchecked<2>();
    for (size_t i = 0; i < static_cast<size_t>(buf.shape(0)); ++i) {
        out << ">" << headers[i] << "\n";
        for (size_t j = 0; j < static_cast<size_t>(buf.shape(1)); ++j) {
            out << buf(i, j);
        }
        out << "\n";
    }

    return true;
}

bool MSAConverter::validate_alignment(const py::array_t<char>& matrix) {
    if (matrix.ndim() != 2) {
        return false;
    }

    auto buf = matrix.unchecked<2>();
    const auto rows = buf.shape(0);
    const auto cols = buf.shape(1);

    for (ssize_t i = 0; i < rows; ++i) {
        for (ssize_t j = 0; j < cols; ++j) {
            char c = buf(i, j);
            if (c != '-' && c != 'A' && c != 'T' && c != 'G' && c != 'C' &&
                c != 'a' && c != 't' && c != 'g' && c != 'c') {
                return false;
            }
        }
    }

    return true;
}

size_t MSAConverter::get_alignment_length(const py::array_t<char>& matrix) {
    if (matrix.ndim() != 2) {
        throw std::runtime_error("Matrix must be 2-dimensional");
    }
    return matrix.shape(1);
}

size_t MSAConverter::get_sequence_count(const py::array_t<char>& matrix) {
    if (matrix.ndim() != 2) {
        throw std::runtime_error("Matrix must be 2-dimensional");
    }
    return matrix.shape(0);
}

} // namespace parsers
} // namespace dnaflex
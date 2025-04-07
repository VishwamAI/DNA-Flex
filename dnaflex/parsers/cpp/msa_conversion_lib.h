#pragma once

#include <string>
#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "fasta_iterator_lib.h"

namespace py = pybind11;

namespace dnaflex {
namespace parsers {

class MSAConverter {
public:
    MSAConverter();
    ~MSAConverter();

    py::array_t<char> fasta_to_matrix(const std::string& filename);
    bool matrix_to_fasta(const py::array_t<char>& matrix,
                        const std::vector<std::string>& headers,
                        const std::string& output_file);

    static bool validate_alignment(const py::array_t<char>& matrix);
    static size_t get_alignment_length(const py::array_t<char>& matrix);
    static size_t get_sequence_count(const py::array_t<char>& matrix);

private:
    std::unique_ptr<FastaIterator> fasta_iter_;
    size_t max_seq_len_;
    std::vector<std::string> sequences_;
    std::vector<std::string> headers_;
};

} // namespace parsers
} // namespace dnaflex
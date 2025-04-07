#pragma once

#include <string>
#include <vector>
#include "fasta_iterator_lib.h"

namespace dnaflex {
namespace parsers {

class MSAConverter {
public:
    MSAConverter();
    ~MSAConverter();

    // Convert FASTA alignment to matrix format
    std::vector<std::vector<char>> fasta_to_matrix(const std::string& filename);
    
    // Convert matrix back to FASTA format
    bool matrix_to_fasta(const std::vector<std::vector<char>>& matrix,
                        const std::vector<std::string>& headers,
                        const std::string& output_file);

    // Utility functions
    static bool validate_alignment(const std::vector<std::vector<char>>& matrix);
    static size_t get_alignment_length(const std::vector<std::vector<char>>& matrix);
    static size_t get_sequence_count(const std::vector<std::vector<char>>& matrix);

private:
    bool check_sequence_lengths(const std::vector<FastaEntry>& entries);
};

} // namespace parsers
} // namespace dnaflex
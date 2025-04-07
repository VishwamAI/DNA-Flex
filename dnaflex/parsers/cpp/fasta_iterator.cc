#include "fasta_iterator_lib.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace dnaflex {
namespace parsers {

FastaIterator::FastaIterator(const std::string& filename) 
    : file_(std::make_unique<std::ifstream>(filename)), 
      valid_(false) {
    if (!file_->is_open()) {
        throw std::runtime_error("Could not open FASTA file: " + filename);
    }
    valid_ = true;
    start_pos_ = file_->tellg();
}

FastaIterator::~FastaIterator() = default;

bool FastaIterator::next(FastaEntry& entry) {
    if (!valid_) {
        return false;
    }

    std::string line;
    bool found_header = false;
    entry.sequence.clear();

    while (std::getline(*file_, line)) {
        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        // Check for header line
        if (line[0] == '>') {
            if (!found_header) {
                entry.header = line.substr(1);
                found_header = true;
            } else {
                // We've found the next sequence header, rewind and break
                file_->seekg(-static_cast<long>(line.length() + 1), std::ios::cur);
                break;
            }
        } else if (found_header) {
            // Append sequence data
            entry.sequence += cleanup_sequence(line);
        }
    }

    return found_header && !entry.sequence.empty();
}

void FastaIterator::reset() {
    if (valid_) {
        file_->clear();
        file_->seekg(start_pos_);
    }
}

bool FastaIterator::is_valid() const {
    return valid_;
}

std::string FastaIterator::cleanup_sequence(const std::string& seq) {
    std::string cleaned;
    cleaned.reserve(seq.length());
    
    for (char c : seq) {
        if (!std::isspace(c)) {
            cleaned += std::toupper(c);
        }
    }
    
    return cleaned;
}

} // namespace parsers
} // namespace dnaflex
#include "fasta_iterator_lib.h"
#include <algorithm>
#include <cctype>

namespace dnaflex {
namespace parsers {

FastaIterator::FastaIterator(const std::string& filename) 
    : valid_(false) {
    file_ = std::make_unique<std::ifstream>(filename);
    valid_ = file_->is_open();
    if (valid_) {
        start_pos_ = file_->tellg();
    }
}

FastaIterator::~FastaIterator() {
    if (file_) {
        file_->close();
    }
}

bool FastaIterator::is_valid() const {
    return valid_;
}

void FastaIterator::reset() {
    if (valid_) {
        file_->clear();
        file_->seekg(start_pos_);
    }
}

bool FastaIterator::next(FastaEntry& entry) {
    if (!valid_ || file_->eof()) {
        return false;
    }
    return read_entry(entry);
}

bool FastaIterator::read_entry(FastaEntry& entry) {
    std::string line;
    entry.header.clear();
    entry.sequence.clear();

    // Find the next header
    while (std::getline(*file_, line)) {
        if (!line.empty() && line[0] == '>') {
            entry.header = line.substr(1);
            break;
        }
    }

    if (entry.header.empty()) {
        return false;
    }

    // Read sequence lines until next header or EOF
    while (std::getline(*file_, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            file_->seekg(-static_cast<int>(line.length()) - 1, std::ios::cur);
            break;
        }
        entry.sequence += cleanup_sequence(line);
    }

    return !entry.sequence.empty();
}

std::string FastaIterator::cleanup_sequence(const std::string& seq) {
    std::string cleaned;
    cleaned.reserve(seq.length());
    std::copy_if(seq.begin(), seq.end(), std::back_inserter(cleaned),
                 [](char c) { return !std::isspace(c); });
    return cleaned;
}

} // namespace parsers
} // namespace dnaflex
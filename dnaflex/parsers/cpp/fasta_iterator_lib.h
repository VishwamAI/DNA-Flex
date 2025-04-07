#pragma once

#include <string>
#include <fstream>
#include <memory>
#include <vector>

namespace dnaflex {
namespace parsers {

struct FastaEntry {
    std::string header;
    std::string sequence;
};

class FastaIterator {
public:
    FastaIterator(const std::string& filename);
    ~FastaIterator();

    bool next(FastaEntry& entry);
    void reset();
    bool is_valid() const;

private:
    std::unique_ptr<std::ifstream> file_;
    bool valid_;
    std::streampos start_pos_;

    bool read_entry(FastaEntry& entry);
    static std::string cleanup_sequence(const std::string& seq);
};

} // namespace parsers
} // namespace dnaflex
#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <pybind11/numpy.h>

namespace dnaflex {
namespace parsers {

struct CifEntry {
    std::string id;
    std::string type;
    std::vector<std::string> values;
};

class CifDictionary {
public:
    CifDictionary();
    ~CifDictionary();

    bool parse(const std::string& filename);
    const CifEntry* get_entry(const std::string& id) const;
    std::vector<std::string> get_categories() const;

private:
    std::unordered_map<std::string, CifEntry> entries_;
    bool parse_block(const std::string& block);
    bool parse_entry(const std::string& entry_text);
};

} // namespace parsers
} // namespace dnaflex
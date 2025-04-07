#include "cif_dict_lib.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace dnaflex {
namespace parsers {

CifDictionary::CifDictionary() {}
CifDictionary::~CifDictionary() {}

bool CifDictionary::parse(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Split into data blocks
    size_t pos = 0;
    while ((pos = content.find("data_", pos)) != std::string::npos) {
        size_t end = content.find("data_", pos + 1);
        if (end == std::string::npos) {
            end = content.length();
        }
        std::string block = content.substr(pos, end - pos);
        if (!parse_block(block)) {
            return false;
        }
        pos = end;
    }

    return true;
}

bool CifDictionary::parse_block(const std::string& block) {
    std::stringstream ss(block);
    std::string line;
    std::string current_entry;
    
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        if (line[0] == '_') {
            if (!current_entry.empty()) {
                if (!parse_entry(current_entry)) {
                    return false;
                }
            }
            current_entry = line;
        } else {
            current_entry += "\n" + line;
        }
    }

    if (!current_entry.empty()) {
        if (!parse_entry(current_entry)) {
            return false;
        }
    }

    return true;
}

bool CifDictionary::parse_entry(const std::string& entry_text) {
    std::stringstream ss(entry_text);
    std::string first_line;
    std::getline(ss, first_line);

    size_t dot_pos = first_line.find('.');
    if (dot_pos == std::string::npos) {
        return false;
    }

    CifEntry entry;
    entry.id = first_line.substr(1, dot_pos - 1);
    entry.type = first_line.substr(dot_pos + 1);

    std::string line;
    while (std::getline(ss, line)) {
        if (!line.empty() && line[0] != '#') {
            entry.values.push_back(line);
        }
    }

    entries_[entry.id] = entry;
    return true;
}

const CifEntry* CifDictionary::get_entry(const std::string& id) const {
    auto it = entries_.find(id);
    return it != entries_.end() ? &it->second : nullptr;
}

std::vector<std::string> CifDictionary::get_categories() const {
    std::vector<std::string> categories;
    for (const auto& entry : entries_) {
        if (std::find(categories.begin(), categories.end(), entry.first) == categories.end()) {
            categories.push_back(entry.first);
        }
    }
    return categories;
}

} // namespace parsers
} // namespace dnaflex
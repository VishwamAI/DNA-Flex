#pragma once

#include <pybind11/pybind11.h>
#include "fasta_iterator_lib.h"

namespace py = pybind11;

namespace dnaflex {
namespace parsers {

void init_fasta_iterator(py::module_& m);

} // namespace parsers
} // namespace dnaflex
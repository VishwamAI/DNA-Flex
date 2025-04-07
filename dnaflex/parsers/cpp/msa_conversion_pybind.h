#pragma once

#include <pybind11/pybind11.h>
#include "msa_conversion_lib.h"

namespace py = pybind11;

namespace dnaflex {
namespace parsers {

void init_msa_conversion(py::module_& m);

} // namespace parsers
} // namespace dnaflex
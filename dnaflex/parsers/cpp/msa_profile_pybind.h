#pragma once

#include <pybind11/pybind11.h>
#include "msa_profile_lib.h"

namespace py = pybind11;

namespace dnaflex {
namespace parsers {

void init_msa_profile(py::module_& m);

} // namespace parsers
} // namespace dnaflex